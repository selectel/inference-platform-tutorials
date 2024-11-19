# Развертывание модели Python на Triton Inference Server

Python Backend позволяет создавать кастомные модели на Python, которые могут выполнять произвольные вычисления. В этом примере мы создадим модель, которая принимает два числа и возвращает их сумму и разность.

## Подготовка модели

1. Создайте директорию для модели:
   ```bash
   mkdir -p model_repository/add_sub/1
   ```

2. Поместите `model.py` и `config.pbtxt` в соответствующие директории.

### Загрузка в S3 хранилище

1. Установите `rclone` и создайте конфигурацию для доступа к S3:
   ```bash
   apt install rclone -y
   ```

2. Создайте конфигурационный файл для `rclone`:
   ```
   [selectel]
   provider = other
   env_auth = false
   access_key_id =
   secret_access_key =
   region = ru-1
   endpoint = s3.ru-1.storage.selcloud.ru
   ```

3. Склонируйте модель в локальное хранилище и загрузите её в S3:
   ```bash
   rclone copy model_repository/ selectel:<bucket_name>/model_repository
   ```

## Настройка инфраструктуры

1. Создайте неймспейс и включите istio-injection:
   ```bash
   kubectl create namespace triton-python
   kubectl label namespace triton-python istio-injection=enabled
   ```

2. Используйте `values` для сценария `demo/base_scenario.yaml`, предварительно скорректировав креды S3:

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

3. Добавьте чарты из нашего харбор:
   ```bash
   helm repo add mldp https://repo.mlops.selectel.ru/chartrepo/mldp
   ```

4. Установите чарт:
   ```bash
   helm upgrade --install -f base_scenario.yaml --namespace triton-python triton-python mldp/triton-inference-server 
   ```

## Пример реализации клиента с использованием python

Для отправки запроса на Triton Inference Server используйте следующий пример с `python`:

```python
import tritonclient.http as httpclient
import numpy as np

# Установите URL вашего Triton Inference Server
url = "<ваш_triton_server_url>"

# Создайте клиента
client = httpclient.InferenceServerClient(url=url)

# Подготовьте входные данные
input0_data = np.array([10.0], dtype=np.float32)
input1_data = np.array([4.0], dtype=np.float32)

# Создайте входные объекты
inputs = [
    httpclient.InferInput("INPUT0", input0_data.shape, "FP32"),
    httpclient.InferInput("INPUT1", input1_data.shape, "FP32")
]
inputs[0].set_data_from_numpy(input0_data)
inputs[1].set_data_from_numpy(input1_data)

# Создайте выходные объекты
outputs = [
    httpclient.InferRequestedOutput("ADD_OUTPUT"),
    httpclient.InferRequestedOutput("SUB_OUTPUT")
]

# Выполните инференс
response = client.infer(model_name="add_sub", inputs=inputs, outputs=outputs)

# Получите результаты
add_output = response.as_numpy("ADD_OUTPUT")
sub_output = response.as_numpy("SUB_OUTPUT")
print("Сумма:", add_output[0])
print("Разность:", sub_output[0])
```

Этот пример показывает, как использовать `python` для отправки запроса на сервер Triton. Убедитесь, что вы заменили `INFERENCE_URL` на фактический URL вашего сервера Triton и предоставили реальные данные для инференса. 