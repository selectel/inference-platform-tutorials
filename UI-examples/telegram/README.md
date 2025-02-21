# Telegram бот для распознавания речи

Этот бот позволяет пользователям отправлять голосовые сообщения и аудиофайлы для распознавания речи с помощью модели Whisper, развернутой в инференс платформе.

## Возможности

- Обработка голосовых сообщений из Telegram
- Поддержка загрузки аудиофайлов
- Автоматическая конвертация аудио в формат WAV с нужными параметрами
- Взаимодействие с Triton Inference Server для распознавания речи
- Поддержка различных языков распознавания через настройку промпта
- Обработка ошибок и информативные сообщения пользователю

## Требования

- Python 3.10+
- FFmpeg для конвертации аудио
- Зависимости из requirements.txt:
  - python-telegram-bot==20.3
  - tritonclient[http]==2.31.0
  - soundfile==0.11.0

## Установка и запуск

### Используя Docker

1. Соберите Docker образ:
```bash
docker build -t telegram-bot .
```

2. Запустите контейнер с необходимыми переменными окружения:
```bash
docker run -d \
  -e TELEGRAM_BOT_TOKEN=<your_token> \
  -e INFERENCE_URL=<triton_server_url> \
  -e WHISPER_PROMPT="<|startoftranscript|><ru>" \
  -e BASIC_AUTH_TOKEN=<auth_token> \
  telegram-bot
```

### Локальный запуск

1. Установите зависимости:
```bash
pip install -r requirements.txt
```

2. Установите FFmpeg:
```bash
apt-get update && apt-get install -y ffmpeg
```

3. Настройте переменные окружения:
```bash
export TELEGRAM_BOT_TOKEN=<your_token>
export INFERENCE_URL=<triton_server_url>
export WHISPER_PROMPT="<|startoftranscript|><ru>"
export BASIC_AUTH_TOKEN=<auth_token>
```

4. Запустите бота:
```bash
python bot.py
```

## Использование

1. Найдите бота в Telegram по его имени
2. Отправьте команду `/start` для начала работы
3. Отправьте голосовое сообщение или аудиофайл
4. Бот вернет распознанный текст

## Структура проекта

```
telegram/
├── Dockerfile          # Конфигурация Docker образа
├── requirements.txt    # Зависимости Python
├── bot.py             # Основной код бота
└── README.md          # Документация
```

## Переменные окружения

- `TELEGRAM_BOT_TOKEN` - токен вашего Telegram бота
- `INFERENCE_URL` - URL Triton сервера с моделью Whisper
- `WHISPER_PROMPT` - промпт для модели Whisper (например, для указания языка)
- `BASIC_AUTH_TOKEN` - токен для basic auth к Triton серверу

## Обработка ошибок

Бот включает обработку различных ошибок:
- Неверный формат аудио
- Ошибки при конвертации
- Проблемы с подключением к Triton серверу
- Ошибки распознавания

При возникновении ошибки пользователь получит информативное сообщение о проблеме. 