# Развертывание модели OpenVINO на Triton Inference Server\
OpenVINO (Open Visual Inference and Neural Network Optimization) — это набор инструментов от Intel, предназначенный для оптимизации и развертывания моделей глубокого обучения. OpenVINO позволяет ускорять инференс на различных аппаратных платформах, включая процессоры Intel, интегрированные графические процессоры и FPGA, обеспечивая высокую производительность и эффективность.

## Подготовка модели

1. Вы можете взять модель из примера onnx в директории [onnx](`../ONNX`).
2. и заменить конфигурационный файл `config.pbtxt` на соответствующий для OpenVINO.
```
name: "densenet_onnx"
backend: "openvino"
default_model_filename: "model.onnx"
```

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
   kubectl create namespace triton-onnx
   kubectl label namespace triton-onnx istio-injection=enabled
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
   helm repo add mldp https://repo.mlops.selcloud.ru/chartrepo/mldp
   ```

4. Установите чарт:
   ```bash
   helm upgrade --install -f base_scenario.yaml --namespace triton-onnx triton-onnx mldp/triton-inference-server 
   ```

## Пример реализации клиента с использованием curl

Установите зависимости и загрузите пример изображения для тестирования инференса.

```bash
docker run -it --rm --net=host -v ${PWD}:/workspace/ nvcr.io/nvidia/tritonserver:<yy.mm>-py3-sdk bash
pip install torchvision

wget -O img1.jpg "https://www.hakaimagazine.com/wp-content/uploads/header-gulf-birds.jpg"
```

Создание клиента требует выполнения трех основных шагов. Во-первых, мы устанавливаем соединение с Triton Inference Server.

```python
client = httpclient.InferenceServerClient(url="<triton_server_url>")
```

Во-вторых, мы указываем имена входного и выходного слоя(ев) нашей модели, а также описываем форму и тип данных ожидаемого входа.

```python
inputs = httpclient.InferInput("data_0", transformed_img.shape, datatype="FP32")
inputs.set_data_from_numpy(transformed_img, binary_data=True)

outputs = httpclient.InferRequestedOutput("fc6_1", binary_data=True, class_count=1000)
```

Наконец, мы отправляем запрос на инференс в Triton Inference Server.

```python
# Запрос к серверу
results = client.infer(model_name="densenet_onnx", inputs=[inputs], outputs=[outputs])
inference_output = results.as_numpy('fc6_1').astype(str)

print(np.squeeze(inference_output)[:5])
```

Вывод должен выглядеть следующим образом:

```
['11.549026:92' '11.232335:14' '7.528014:95' '6.923391:17' '6.576575:88']
```

Формат вывода: <confidence_score>:<classification_index>. Чтобы узнать, как сопоставить эти значения с именами меток и получить дополнительную информацию, обратитесь к нашей документации. Код клиента выше доступен в файле `client.py`.