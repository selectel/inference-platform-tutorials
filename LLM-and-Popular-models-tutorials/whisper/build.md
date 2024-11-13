# Инструкция по сборке Whisper

Эта инструкция основана на [репозитории Sherpa](https://github.com/k2-fsa/sherpa/tree/master/triton/whisper). Для начала необходимо склонировать репозиторий:

```sh
git clone https://github.com/k2-fsa/sherpa.git
```
### 0. Подготовка окружения

Билд образа может осуществляться на любой машине, поддерживающей GPU и где настроен docker для работы с GPU.
Подготовка окружения описана в [utils/docker-gpu.md](../utils/docker-gpu.md).

## Сборка Docker-образа

Для сборки Docker-образа выполните следующую команду:

```bash
docker build -t <tag> -f Dockerfile .
```

## Сборка модели Whisper

Запустите контейнер с Whisper:

```sh
docker run -it --name "whisper-server" --gpus all --net host -v $PWD:/workspace --shm-size=2g soar97/triton-whisper:24.05
```

### Подготовка `model_repo_whisper_trtllm`

Перейдите в директорию с Whisper и выполните скрипт сборки:

```sh
cd sherpa/triton/whisper
cd ./whisper_large_v3_trtllm_triton && bash ./build_whisper_fp16.sh
cp -r /workspace/TensorRT-LLM/examples/whisper/whisper_large_v3 ./model_repo_whisper_trtllm/whisper/1/
```

> **Важно:** Папку `whisper_large_v3` необходимо копировать как абсолютный путь, так как в ней хранятся веса энкодеров и декодеров.

### Структура модели для S3

Модель должна быть загружена в S3 в следующей структуре:

```
model_repo_whisper_trtllm
└── whisper
    ├── 1
    │   ├── fbank.py
    │   ├── mel_filters.npz
    │   ├── model.py
    │   ├── multilingual.tiktoken
    │   ├── tokenizer.py
    │   ├── whisper_large_v3 
    │   │   ├── encoder ...
    │   │   ├── decoder ...
    │   └── whisper_trtllm.py
    └── config.pbtxt
```

Для загрузки модели в S3 можно использовать [Rclone](https://docs.selectel.ru/en/cloud/object-storage/tools/rclone/).



