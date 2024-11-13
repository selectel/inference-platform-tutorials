# Деплой Whisper

## Подготовка образа и модели
Подготовка образа и модели описана в [build.md](build.md).

## Деплой инференса
Для развертывания инференса необходимо работать в кластере, где настроена инференс платформа.

Создайте неймспейс и включите istio-injection:
```bash
kubectl create namespace whisper-demo
kubectl label namespace whisper-demo istio-injection=enabled
```

Используйте `values` для сценария `demo/base_scenario.yaml`, предварительно скорректировав креды S3:

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
  imageName: # Укажите имя образа, который вы собрали в build.md
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

Добавьте чарты из вашего репозитория:
```bash
helm repo add mldp https://repo.mlops.selectel.ru/chartrepo/mldp
```

Установите чарт:
```bash
helm upgrade --install -f values.yaml --namespace whisper-demo whisper-demo mldp/triton-inference-server 
```

Подключитесь к Grafana (через port-forward) и просмотрите дашборд:

```bash
kubectl --namespace inferp-platform port-forward svc/grafana 3000
```

Отправьте запрос на выполнение инференса с помощью скрипта [client.py](client/client.py)