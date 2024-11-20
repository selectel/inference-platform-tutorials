# Деплой модели vLLM в Triton

Следующий туториал демонстрирует, как развернуть простую модель
[gpt2](https://huggingface.co/openai-community/gpt2) на
Triton Inference Server, используя
[Python-backend](https://github.com/triton-inference-server/backend/blob/main/docs/python_based_backends.md#python-based-backends)
бэкенд [vLLM](https://github.com/triton-inference-server/vllm_backend/tree/main).


## Шаг 1: Подготовьте репозиторий модели

Для использования Triton нам нужно создать репозиторий модели. В этом туториале мы будем использовать репозиторий модели, предоставленный в папке [samples](https://github.com/triton-inference-server/vllm_backend/tree/main/samples) репозитория [vllm_backend](https://github.com/triton-inference-server/vllm_backend/tree/main).

В директории `model_repository` создайте директорию `vllm_model` и скопируйте файлы [`model.json`](./model_repository/vllm_model/1/model.json) и [`config.pbtxt`](./model_repository/vllm_model/config.pbtxt) в неё.


Репозиторий модели должен выглядеть следующим образом:
```
model_repository/
└── vllm_model
    ├── 1
    │   └── model.json
    └── config.pbtxt
```

Обратите внимание, что контейнер Triton с vLLM был введен начиная с релиза 23.10.

Этот файл можно изменить для предоставления дополнительных настроек движку vLLM. См. vLLM [AsyncEngineArgs](https://github.com/vllm-project/vllm/blob/32b6816e556f69f1672085a6267e8516bcb8e622/vllm/engine/arg_utils.py#L165) и [EngineArgs](https://github.com/vllm-project/vllm/blob/32b6816e556f69f1672085a6267e8516bcb8e622/vllm/engine/arg_utils.py#L11) для поддерживаемых пар ключ-значение. Пакетная обработка и страничное внимание обрабатываются движком vLLM.

Для поддержки нескольких GPU можно указать такие параметры EngineArgs, как `tensor_parallel_size`, в [`model.json`](https://github.com/triton-inference-server/vllm_backend/blob/main/samples/model_repository/vllm_model/1/model.json).

*Примечание*: По умолчанию vLLM жадно использует до 90% памяти GPU. Этот туториал изменяет это поведение, устанавливая `gpu_memory_utilization` на 50%. Вы можете настроить это поведение, используя поля, такие как `gpu_memory_utilization`, и другие настройки в [`model.json`](https://github.com/triton-inference-server/vllm_backend/blob/main/samples/model_repository/vllm_model/1/model.json).

Прочтите документацию в [`model.py`](https://github.com/triton-inference-server/vllm_backend/blob/main/src/model.py), чтобы понять, как настроить этот пример для вашего случая использования.

## Шаг 2: Запуск Triton Inference Server в Inference платформе

На S3 скачиваем модель `gpt2` из нашего репозитория:

```bash
apt install rclone -y
mkdir model_repository
cp -r vLLM/model_repository model_repository
```

С помощью утилиты [rclone](https://docs.selectel.ru/cloud/object-storage/tools/rclone/) готовим конфиг следующего содержания:

```
[selectel]
provider = other
env_auth = false
access_key_id =
secret_access_key =
region = ru-1
endpoint = s3.ru-1.storage.selcloud.ru
```

Склонируем себе локально модель и загрузим её в бакет S3:

```bash
rclone copy model_repository/ selectel:<bucket_name>/model_repository
```

Также на SFS создаем папку для хранения кеша модели:

```bash
sudo mkdir -p /mnt/nfs && sudo mount -vt nfs "<sfs ip>:<sfs mount path>" /mnt/nfs
mkdir -p /mnt/nfs/hf_cache
```

Наша инфраструктура готова.

## Базовый сценарий
Создаем неймспейс и энейблим istio-injection:
```bash
kubectl create namespace triton-demo-1
kubectl label namespace triton-demo-1 istio-injection=enabled
```

Используем `values` для сценария `demo/base_scenario.yaml`, предварительно скорректировав креды S3:

```yaml
tags:
  autoscaling: false
  traefikBalancing: false
  istioGateway: true
  canary: false
  sfs: false
  s3: true
  istioBasicAuth:
    main:
      enable: true
      passwordBase64: # Добавьте base64-кодированный пароль

main:
  imageName: # Укажите имя образа, например, repo.mlops.selcloud.ru/mldp/triton_transformer_server:24.05-zstd
  numGpus: 1
  environment:
    TRITON_AWS_MOUNT_DIRECTORY: # Укажите путь к директории монтирования AWS, например, /opt/tritonserver

  serverArgs:
    - '--model-repository=s3://# Укажите URL репозитория модели, например, https://s3.ru-1.storage.selcloud.ru:443/<bucket_name>/model_repository'
    - '--log-verbose=1'
  nodeSelector:
    demo: "base"

secret:
  s3:
    region: # Укажите регион, например, ru-1
    id: # Укажите ID
    key: # Укажите ключ
```

Добавим чарты из нашего харбор:
```bash
helm repo add mldp https://repo.mlops.selcloud.ru/chartrepo/mldp
```

И установим чарт:
```bash
helm upgrade --install -f base_scenario.yaml --namespace triton-demo-1 triton-demo-1 mldp/triton-inference-server 
```



## Шаг 3: Использование клиента Triton для отправки первого запроса на инференс

Теперь отправим запрос на инференс:
```
curl -X POST $INFERENCE_SERVER_URL/v2/models/vllm_model/generate -d '{"text_input": "What is Triton Inference Server?", "parameters": {"stream": false, "temperature": 0}}'
```