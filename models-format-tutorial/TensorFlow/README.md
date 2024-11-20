# Развертывание модели TensorFlow на Triton Inference Server

TensorFlow — это популярный фреймворк для разработки и обучения моделей глубокого обучения. Triton Inference Server позволяет легко развертывать модели TensorFlow для высокопроизводительного инференса.

## Подготовка модели

1. Скачайте модель ResNet50 в формате SavedModel:
   ```bash
   mkdir -p model_repository/resnet50_tf/1
   wget -O model_repository/resnet50_tf/1/model.savedmodel \
        https://storage.googleapis.com/tfhub-modules/tensorflow/resnet_50/classification/1.tar.gz
   tar -xzf model_repository/resnet50_tf/1/model.savedmodel
   ```

2. Создайте файл конфигурации `config.pbtxt` в директории `model_repository/resnet50_tf/`:
   ```plaintext
   name: "resnet50_tf"
   platform: "tensorflow_savedmodel"
   max_batch_size: 1

   input [
     {
       name: "input_tensor"
       data_type: TYPE_FP32
       dims: [224, 224, 3]
     }
   ]

   output [
     {
       name: "softmax_tensor"
       data_type: TYPE_FP32
       dims: [1000]
     }
   ]

   instance_group [
     {
       kind: KIND_GPU
     }
   ]
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
   kubectl create namespace triton-tensorflow
   kubectl label namespace triton-tensorflow istio-injection=enabled
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
   helm upgrade --install -f base_scenario.yaml --namespace triton-tensorflow triton-tensorflow mldp/triton-inference-server 
   ```

## Пример реализации клиента с использованием python

Для отправки запроса на Triton Inference Server используйте следующий пример с `python`:

```python
import numpy as np
import tritonclient.http as httpclient
from PIL import Image
from torchvision import transforms
from tritonclient.utils import triton_to_np_dtype

# preprocessing function
def preprocess(img_path="img1.jpg"):
    img = Image.open(img_path)
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return preprocess(img).numpy()

transformed_img = preprocess()

# Setting up client
client = httpclient.InferenceServerClient(url="<INFERENCE_URL>")

inputs = httpclient.InferInput("input_tensor", transformed_img.shape, datatype="FP32")
inputs.set_data_from_numpy(transformed_img, binary_data=True)

outputs = httpclient.InferRequestedOutput(
    "softmax_tensor", binary_data=True, class_count=1000
)

# Querying the server
results = client.infer(model_name="resnet50_tf", inputs=[inputs], outputs=[outputs])
inference_output = results.as_numpy("softmax_tensor")
print(inference_output[:5])
```

Этот пример показывает, как использовать `python` для отправки запроса на сервер Triton. Убедитесь, что вы заменили `INFERENCE_URL` на фактический URL вашего сервера Triton и предоставили реальные данные для инференса. 