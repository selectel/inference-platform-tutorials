import os

# Определение пути к кешу для модели Hugging Face.
# Это необходимо для избежания повторной загрузки модели при каждом запуске.
os.environ["TRANSFORMERS_CACHE"] = "/opt/tritonserver/model_repository/falcon7b/hf_cache"

import json
import numpy as np
import torch
import transformers  # библиотека Hugging Face для работы с моделями NLP
import triton_python_backend_utils as pb_utils  # утилиты Triton для взаимодействия с Python backend


class TritonPythonModel:
    def initialize(self, args):
        # Инициализация логгера для записи сообщений во время работы модели
        self.logger = pb_utils.Logger

        # Загрузка конфигурации модели из аргументов (json формат)
        self.model_config = json.loads(args["model_config"])

        # Извлечение параметров модели из конфигурации Triton (если указаны)
        self.model_params = self.model_config.get("parameters", {})

        # Определение модели по умолчанию, если в конфигурации не указана другая модель
        default_hf_model = "tiiuae/falcon-7b"  # модель Falcon 7B из Hugging Face
        default_max_gen_length = "15"  # максимальная длина сгенерированного текста по умолчанию

        # Проверка, указана ли в конфигурации пользовательская модель Hugging Face
        hf_model = self.model_params.get("huggingface_model", {}).get(
            "string_value", default_hf_model
        )

        # Проверка, указана ли в конфигурации максимальная длина сгенерированного текста
        self.max_output_length = int(
            self.model_params.get("max_output_length", {}).get(
                "string_value", default_max_gen_length
            )
        )

        # Логирование информации о выбранной модели и максимальной длине сгенерированного текста
        self.logger.log_info(f"Max sequence length: {self.max_output_length}")
        self.logger.log_info(f"Loading HuggingFace model: {hf_model}...")

        # Загрузка токенизатора модели из Hugging Face для обработки входного текста
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(hf_model)

        # Загрузка пайплайна для генерации текста с использованием модели и токенизатора
        # torch_dtype используется для ускорения вычислений (в данном случае fp16)
        # device_map="auto" позволяет автоматически выбрать устройство (CPU/GPU) для модели
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=hf_model,
            torch_dtype=torch.float16,
            tokenizer=self.tokenizer,
            device_map="auto",
        )

        # Установка специального токена конца предложения (EOS) как токена для паддинга
        self.pipeline.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def execute(self, requests):
        prompts = []  # список для хранения входных данных
        # Обработка запросов (входной текст передается в формате тензоров)
        for request in requests:
            # Получение текстового ввода из запроса
            input_tensor = pb_utils.get_input_tensor_by_name(request, "text_input")

            # Проверка, является ли вход многомерным (например, если запросы динамически батчатся)
            multi_dim = input_tensor.as_numpy().ndim > 1
            if not multi_dim:
                # Если вход одномерный, декодируем строку текста
                prompt = input_tensor.as_numpy()[0].decode("utf-8")
                self.logger.log_info(f"Generating sequences for text_input: {prompt}")
                prompts.append(prompt)
            else:
                # Если вход многомерный, обрабатываем каждый запрос отдельно
                num_prompts = input_tensor.as_numpy().shape[0]
                for prompt_index in range(0, num_prompts):
                    prompt = input_tensor.as_numpy()[prompt_index][0].decode("utf-8")
                    prompts.append(prompt)

        # Определение размера батча (количества запросов) для генерации
        batch_size = len(prompts)
        return self.generate(prompts, batch_size)

    def generate(self, prompts, batch_size):
        # Генерация текста на основе входных запросов с использованием модели
        sequences = self.pipeline(
            prompts,
            max_length=self.max_output_length,  # максимальная длина выходного текста
            pad_token_id=self.tokenizer.eos_token_id,  # токен для паддинга
            batch_size=batch_size,  # размер батча
        )

        responses = []  # список для хранения ответов
        texts = []  # список для хранения сгенерированных текстов
        for i, seq in enumerate(sequences):
            output_tensors = []
            text = seq[0]["generated_text"]  # извлечение сгенерированного текста
            texts.append(text)
            tensor = pb_utils.Tensor("text_output", np.array(texts, dtype=np.object_))  # создание тензора для вывода
            output_tensors.append(tensor)
            responses.append(pb_utils.InferenceResponse(output_tensors=output_tensors))  # создание ответа

        return responses

    def finalize(self):
        # Очистка ресурсов при завершении работы модели
        print("Cleaning up...")
