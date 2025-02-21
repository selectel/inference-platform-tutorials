import logging
from telegram import Update, InputFile
from telegram.ext import Application, CommandHandler, MessageHandler, filters
import requests
import os

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


# Функция для отправки аудио на внешний сервис
def send_audio_to_service(audio_path):
    logger.info(f"Отправка аудио {audio_path} на внешний сервис.")
    file = {'file': open(audio_path, 'rb')}
    response = requests.post(url=os.environ["INFERENCE_URL"], files=file)  # Изменён порт
    if response.status_code == 200:
        logger.info("Изображение успешно получено от сервиса.")
    else:
        logger.error(f"Ошибка при отправке аудио: {response.status_code}")

    # Сохраняем изображение, полученное в ответ
    image_path = '../image.png'
    with open(image_path, 'wb') as f:
        f.write(response.content)

    return image_path


# Обработчик команды /start
async def start(update: Update, context) -> None:
    logger.info(f"Пользователь {update.effective_user.username} начал взаимодействие с ботом.")
    await update.message.reply_text(
        "Привет! Выберите действие:\n1. Отправить аудио."
    )


# Обработчик получения аудиофайла от пользователя
async def handle_audio(update: Update, context) -> None:
    user = update.effective_user.username
    logger.info(f"Получен аудиофайл от пользователя {user}.")

    audio_file = await update.message.audio.get_file()

    # Сохраняем аудио
    audio_path = './audio_file.m4a'
    await audio_file.download_to_drive(audio_path)
    logger.info(f"Аудиофайл сохранён: {audio_path}.")

    # Отправляем аудио на внешний сервис
    image_path = send_audio_to_service(audio_path)

    # Отправляем картинку пользователю
    with open(image_path, 'rb') as img:
        await update.message.reply_photo(photo=img)
        logger.info(f"Изображение отправлено пользователю {user}.")

    # Удаляем временные файлы после использования
    os.remove(audio_path)
    os.remove(image_path)
    logger.info(f"Временные файлы {audio_path} и {image_path} удалены.")


# Основная функция для запуска бота
def main() -> None:
    # Токен бота
    token = os.environ["TELEGRAM_BOT_TOKEN"]

    # Создание объекта приложения
    application = Application.builder().token(token).build()

    # Регистрация обработчиков
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.AUDIO, handle_audio))

    logger.info("Бот запущен и ожидает сообщений.")

    # Запуск бота
    application.run_polling()


if __name__ == '__main__':
    main()
