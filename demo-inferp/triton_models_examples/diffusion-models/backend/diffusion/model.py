import json
import os
import shutil
import sys

import numpy as np
import torch
from cuda import cudart

# Определение местоположения файла текущего скрипта
file_location = os.path.dirname(os.path.realpath(__file__))

# Вставка пути к модулям модели Stable Diffusion
sys.path.insert(0, os.path.join(file_location, "Diffusion"))

import triton_python_backend_utils as pb_utils
from Diffusion.stable_diffusion_pipeline import StableDiffusionPipeline  # импорт пайплайна для генерации изображений
from Diffusion.utilities import PIPELINE_TYPE  # импорт утилит для работы с типами пайплайнов

class TritonPythonModel:
    _KNOWN_VERSIONS = {"1.5": PIPELINE_TYPE.TXT2IMG, "xl-1.0": PIPELINE_TYPE.XL_BASE}  # поддерживаемые версии Stable Diffusion

    def _set_defaults(self):
        # Установка значений по умолчанию для модели
        self._batch_size = 1  # размер батча
        self._onnx_opset = 18  # версия ONNX opset
        self._image_height = 512  # высота изображения по умолчанию
        self._image_width = 512  # ширина изображения по умолчанию
        self._seed = None  # семя для генерации случайных чисел (опционально)
        self._version = "1.5"  # версия модели по умолчанию
        self._scheduler = None  # планировщик (если не указан, по умолчанию не используется)
        self._steps = 30  # количество шагов диффузии
        self._force_engine_build = False  # нужно ли принудительно пересобрать движок

    def _set_from_parameter(self, parameter, parameters, class_):
        # Функция для установки параметра из конфигурации модели
        value = parameters.get(parameter, None)
        if value is not None:
            value = value["string_value"]
            if value:
                setattr(self, "_" + parameter, class_(value))

    def _set_from_config(self, model_config):
        # Функция для получения параметров модели из конфигурации Triton
        model_config = json.loads(model_config)
        self._batch_size = int(model_config.get("max_batch_size", 1))  # установка размера батча из конфигурации
        if self._batch_size < 1:
            self._batch_size = 1

        config_parameters = model_config.get("parameters", {})

        if config_parameters:
            # Словарь для сопоставления параметров с их типами
            parameter_type_map = {
                "onnx_opset": int,
                "image_height": int,
                "image_width": int,
                "steps": int,
                "seed": int,
                "scheduler": str,
                "guidance_scale": float,
                "version": str,
                "force_engine_build": bool,
            }

            # Установка всех параметров из конфигурации Triton
            for parameter, parameter_type in parameter_type_map.items():
                self._set_from_parameter(parameter, config_parameters, parameter_type)

    def initialize(self, args):
        # Инициализация модели: установка значений по умолчанию и получение параметров из конфигурации
        self._set_defaults()
        self._set_from_config(args["model_config"])

        # Проверка версии Stable Diffusion
        if self._version not in TritonPythonModel._KNOWN_VERSIONS:
            raise Exception(
                f"Invalid Stable Diffusion Version: {self._version}, choices: {list(TritonPythonModel._KNOWN_VERSIONS.keys())}"
            )

        self._model_instance_device_id = int(args["model_instance_device_id"])  # устройство для модели

        # Инициализация пайплайна Stable Diffusion
        self._pipeline = StableDiffusionPipeline(
            pipeline_type=TritonPythonModel._KNOWN_VERSIONS[self._version],  # выбор типа пайплайна (текст-изображение или другое)
            max_batch_size=self._batch_size,  # максимальный размер батча
            use_cuda_graph=True,  # использование CUDA для ускорения
            version=self._version,  # версия модели
            denoising_steps=self._steps,  # количество шагов диффузии
        )

        # Пути для хранения моделей и движков
        model_directory = os.path.join(args["model_repository"], args["model_version"])
        engine_dir = os.path.join(
            model_directory, f"{self._version}-engine-batch-size-{self._batch_size}"
        )
        framework_model_dir = os.path.join(
            model_directory, f"{self._version}-pytorch_model"
        )
        onnx_dir = os.path.join(model_directory, f"{self._version}-onnx")

        # Удаление старых движков при необходимости пересборки
        if self._force_engine_build:
            shutil.rmtree(engine_dir, ignore_errors=True)
            shutil.rmtree(framework_model_dir, ignore_errors=True)
            shutil.rmtree(onnx_dir, ignore_errors=True)

        # Валидация устройства
        if self._model_instance_device_id != 0:
            raise Exception("Only device id 0 is currently supported")

        # Загрузка движков и моделей
        self._pipeline.loadEngines(
            engine_dir,
            framework_model_dir,
            onnx_dir,
            onnx_opset=self._onnx_opset,
            opt_batch_size=self._batch_size,
            opt_image_height=self._image_height,
            opt_image_width=self._image_width,
            static_batch=True,  # статический батчинг для оптимизации
        )

        # Выделение памяти для работы с CUDA
        _, shared_device_memory = cudart.cudaMalloc(
            self._pipeline.calculateMaxDeviceMemory()
        )

        # Активация движков
        self._pipeline.activateEngines(shared_device_memory)

        # Загрузка дополнительных ресурсов для генерации изображений
        self._pipeline.loadResources(
            self._image_height, self._image_width, self._batch_size, seed=self._seed
        )

        # Инициализация логгера Triton
        self._logger = pb_utils.Logger

    def finalize(self):
        # Очистка ресурсов модели при завершении работы
        self._pipeline.teardown()

    def execute(self, requests):
        responses = []
        prompts = []
        negative_prompts = []
        prompts_per_request = []
        image_results = []
        for request in requests:
            # Извлечение текста запроса (prompt) для генерации изображений
            prompt_tensor = pb_utils.get_input_tensor_by_name(
                request, "prompt"
            ).as_numpy()

            for prompt in prompt_tensor:
                prompts.append(prompt[0].decode())

            # Извлечение негативного текста запроса (negative prompt) для управления генерацией
            negative_prompt_tensor = pb_utils.get_input_tensor_by_name(
                request, "negative_prompt"
            )

            # Если негативные промпты не заданы, создаем пустые строки
            if not negative_prompt_tensor:
                negative_prompts.extend([""] * len(prompt_tensor))
            else:
                negative_prompt_tensor = negative_prompt_tensor.as_numpy()
                for negative_prompt in negative_prompt_tensor:
                    negative_prompts.append(negative_prompt[0].decode())
            prompts_per_request.append(len(prompt_tensor))

        # Логирование количества запросов и промптов в батче
        num_requests = len(requests)
        num_prompts = len(prompts)
        remainder = self._batch_size - (num_prompts % self._batch_size)
        self._logger.log_info(f"Client Requests in Batch:{num_requests}")
        self._logger.log_info(f"Prompts in Batch:{num_prompts}")

        # Дополнение до размера батча, если требуется
        if remainder < self._batch_size:
            prompts.extend([""] * remainder)
            negative_prompts.extend([""] * remainder)

        num_prompts = len(prompts)
        for batch in range(0, num_prompts, self._batch_size):
            # Генерация изображений на основе промптов
            (images, walltime_ms) = self._pipeline.infer(
                prompts[batch : batch + self._batch_size],
                negative_prompts[batch : batch + self._batch_size],
                self._image_height,
                self._image_width,
                save_image=False,
            )
            # Обработка изображений для сохранения в правильном формате
            images = (
                ((images + 1) * 255 / 2)
                .clamp(0, 255)
                .detach()
                .permute(0, 2, 3, 1)
                .round()
                .type(torch.uint8)
                .cpu()
                .numpy()
            )
            image_results.extend(images)

        # Создание ответов для каждого запроса
        result_index = 0
        for num_prompts_in_request in prompts_per_request:
            generated_images = []
            for image_result in image_results[
                result_index : result_index + num_prompts_in_request
            ]:
                generated_images.append(image_result)
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor(
                        "generated_image",
                        np.array(generated_images, dtype=np.uint8),
                    )
                ]
            )
            responses.append(inference_response)
            result_index += num_prompts_in_request

        return responses
