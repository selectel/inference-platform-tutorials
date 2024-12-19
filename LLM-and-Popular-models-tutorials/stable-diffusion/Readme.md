# Деплой Stable Diffusion XL

## Подготовка образа и модели
Подготовка образа и модели описана в [build.md](build.md).

## Деплой инференса
Далее необходимо работать в кластере, где настроена инференс платформа.

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
    access_key: # Укажите Access Key
    secret_key: # Укажите Secret Key
```

Добавим чарты из нашего харбор:
```bash
helm repo add mldp https://repo.mlops.selcloud.ru/chartrepo/mldp
```

И установим чарт:
```bash
helm upgrade --install -f values.yaml --namespace triton-demo-1 triton-demo-1 mldp/triton-inference-server 
```

Зайдем в графана (через port-forward), посмотрим на дашборд. 

```bash
kubectl --namespace inferp-platform port-forward svc/grafana 3000
```

Подадим запрос на выполнение инференса:

```bash
export INFERENCE_URL=<взять из grafana>
curl -X POST $INFERENCE_URL -d '{"inputs": [{"name":"prompt","datatype":"TYPE_STRING","shape":[1],"data":["pigeon in new york, realistic, 4k, photograph"]}]}'
```