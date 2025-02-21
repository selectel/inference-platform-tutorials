# Примеры пользовательских интерфейсов для инференс платформы

В этом разделе представлены примеры реализации пользовательских интерфейсов для взаимодействия с моделями, развернутыми в инференс платформе.

## Telegram бот для распознавания речи

[Telegram бот](telegram/README.md) демонстрирует пример интеграции с моделью Whisper для распознавания речи через мессенджер Telegram. Бот принимает голосовые сообщения и аудиофайлы, отправляет их на инференс сервер и возвращает распознанный текст.

### Основные возможности:
- Обработка голосовых сообщений
- Поддержка загрузки аудиофайлов
- Конвертация аудио в нужный формат
- Взаимодействие с Triton Inference Server
- Поддержка различных языков распознавания

### Запуск бота:
```bash
# Сборка Docker образа
docker build -t telegram-bot .

# Запуск контейнера с переменными окружения
docker run -d \
  -e TELEGRAM_BOT_TOKEN=<your_token> \
  -e INFERENCE_URL=<triton_server_url> \
  -e WHISPER_PROMPT="<|startoftranscript|><ru>" \
  -e BASIC_AUTH_TOKEN=<auth_token> \
  telegram-bot
```

## Gradio интерфейс для распознавания речи

[Gradio приложение](gradio/README.md) предоставляет веб-интерфейс для взаимодействия с моделью Whisper. Интерфейс позволяет загружать аудиофайлы, записывать аудио с микрофона или указывать URL аудиофайла для распознавания.

### Основные возможности:
- Загрузка аудиофайлов с диска
- Запись аудио с микрофона
- Загрузка аудио по URL
- Выбор языка распознавания
- Настройка параметров модели
- Отображение информации о процессе распознавания

### Запуск приложения:
```bash
python audio_gradio.py
```

## Развертывание модели Whisper

Для работы интерфейсов требуется развернутая модель Whisper в инференс платформе. Инструкции по развертыванию модели доступны в [документации по Whisper](../LLM-and-Popular-models-tutorials/whisper/Readme.md).

Пример конфигурации для развертывания:

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
  imageName: # Укажите имя образа с whisper
  numGpus: 1
  environment:
    TRITON_AWS_MOUNT_DIRECTORY: # Укажите путь к директории монтирования AWS

  serverArgs:
    - '--model-repository=s3://# Укажите URL репозитория модели'
    - '--log-verbose=1'
  nodeSelector:
    demo: "base"

secret:
  s3:
    region: # Укажите регион
    access_key: # Укажите Access Key
    secret_key: # Укажите Secret Key
```

## Дополнительные материалы

- [Документация Telegram Bot API](https://core.telegram.org/bots/api)
- [Документация Gradio](https://www.gradio.app/docs)
- [Документация Triton Inference Server](https://github.com/triton-inference-server/server)
