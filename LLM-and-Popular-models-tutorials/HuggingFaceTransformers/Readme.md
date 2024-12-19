# Демонстрация работы инференс платформы с моделью Falcon-7B

### Скачивание модели
На S3 скачиваем модель `falcon7b` из нашего репозитория:

```bash
apt install rclone -y
mkdir model_repository
cp -r falcon7b/model_repository model_repository
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
    access_key: # Укажите Access Key
    secret_key: # Укажите Secret Key
```

Добавим чарты из нашего харбор:
```bash
helm repo add mldp https://repo.mlops.selcloud.ru/chartrepo/mldp
```

И установим чарт:
```bash
helm upgrade --install -f base_scenario.yaml --namespace triton-demo-1 triton-demo-1 mldp/triton-inference-server 
```

Зайдем в графана, посмотрим на дашборд. Подадим запрос на выполнение инференса:

```bash
export INFERENCE_URL=<взять из grafana>
curl -X POST $INFERENCE_URL -d '{"inputs": [{"name":"text_input","datatype":"BYTES","shape":[1],"data":["I am going"]}]}'
```