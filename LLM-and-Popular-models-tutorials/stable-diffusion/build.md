# Инструкция по сборке Stable Diffusion

Эта инструкция основана на примере из [репозитория Triton](https://github.com/triton-inference-server/tutorials/tree/main/Triton_Inference_Server_Python_API/examples/rayserve), который оказался нерабочим. Поэтому сборка была выполнена самостоятельно, используя [репозиторий Stable Diffusion](https://github.com/triton-inference-server/tutorials/tree/main/Popular_Models_Guide/StableDiffusion).

## Шаги по сборке

### 0. Подготовка окружения

Билд образа может осуществляться на любой машине, поддерживающей GPU и где настроен docker для работы с GPU.
Подготовка окружения описана в [utils/docker-gpu.md](../utils/docker-gpu.md).

### 1. Сборка Docker-образа Tritonserver для Diffusion

Запустите скрипт сборки:

```bash
./build.sh
```

### 2. Сборка и запуск Stable Diffusion XL

#### Запуск контейнера Tritonserver Diffusion

Следующая команда запускает контейнер и монтирует текущую директорию как `workspace`:

```bash
./run.sh
```

#### Сборка движка Stable Diffusion XL

Выполните команду:

```bash
./scripts/build_models.sh --model stable_diffusion_xl
```

##### Ожидаемый результат

```
 diffusion-models
 |-- stable_diffusion_xl
    |-- 1
    |   |-- xl-1.0-engine-batch-size-1
    |   |-- xl-1.0-onnx
    |   `-- xl-1.0-pytorch_model
    `-- config.pbtxt
```

#### Запуск экземпляра сервера

> **Примечание:** Для демонстрации используется режим управления моделями `EXPLICIT`, чтобы контролировать, какая версия Stable Diffusion загружается. Для производственных развертываний ознакомьтесь с [рекомендациями по безопасному развертыванию][secure_guide] для получения дополнительной информации о рисках, связанных с режимом `EXPLICIT`.

```bash
tritonserver --model-repository diffusion-models --model-control-mode explicit --load-model stable_diffusion_xl
```

##### Ожидаемый результат

```
<SNIP>
I0229 20:22:22.912465 1440 server.cc:676]
+---------------------+---------+--------+
| Model               | Version | Status |
+---------------------+---------+--------+
| stable_diffusion_xl | 1       | READY  |
+---------------------+---------+--------+

<SNIP>
```

### 3. Сборка и запуск Stable Diffusion XL с Ray Serve

После выполнения предыдущих шагов у вас должен быть образ `tritonserver:r24.01-diffusion`.

#### Сборка образа

Используйте [Dockerfile](Dockerfile) для сборки:

```bash
docker build -t <namet:tag> -f Dockerfile .
```

Вы можете загрузить этот образ в удаленный репозиторий.

```bash
docker push -t <namet:tag>
```

#### Загрузка модели в S3

Модель должна быть загружена в S3 в следующей структуре:

```
 diffusion-models
 |-- stable_diffusion_xl
    |-- 1
    |   |-- xl-1.0-engine-batch-size-1
    |   |-- xl-1.0-onnx
    |   `-- xl-1.0-pytorch_model
    `-- config.pbtxt
```

Для загрузки можно использовать [Rclone](https://docs.selectel.ru/en/cloud/object-storage/tools/rclone/).