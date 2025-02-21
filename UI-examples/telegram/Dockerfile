# Используем официальный образ Python
FROM python:3.10-slim

# Устанавливаем зависимости для ffmpeg и других утилит
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файл с зависимостями
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем исходный код приложения
COPY . .

# Указываем команду для запуска бота
CMD ["python", "bot.py"]
