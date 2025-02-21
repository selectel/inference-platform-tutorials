import logging
import os
import tempfile
import time
from datetime import datetime

import numpy as np
import urllib.request
import soundfile
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


# Функция для конвертации в WAV
def convert_to_wav(in_filename: str) -> str:
    out_filename = in_filename + ".wav"
    if '.mp3' in in_filename:
        _ = os.system(f"ffmpeg -y -i '{in_filename}' -acodec pcm_s16le -ac 1 -ar 16000 '{out_filename}' || exit 1")
    else:
        _ = os.system(f"ffmpeg -hide_banner -y -i '{in_filename}' -ar 16000 '{out_filename}' || exit 1")
    return out_filename


# Функция отправки на Triton сервер
def send_whisper(whisper_prompt, wav_path, model_name, triton_client, protocol_client, padding_duration=10):
    waveform, sample_rate = soundfile.read(wav_path)
    assert sample_rate == 16000, f"Only support 16k sample rate, but got {sample_rate}"
    duration = int(len(waveform) / sample_rate)

    samples = np.zeros(
        (
            1,
            padding_duration * sample_rate * ((duration // padding_duration) + 1),
        ),
        dtype=np.float32,
    )
    samples[0, : len(waveform)] = waveform

    lengths = np.array([[len(waveform)]], dtype=np.int32)

    inputs = [
        protocol_client.InferInput(
            "WAV", samples.shape, np_to_triton_dtype(samples.dtype)
        ),
        protocol_client.InferInput(
            "TEXT_PREFIX", [1, 1], "BYTES"
        ),
    ]
    inputs[0].set_data_from_numpy(samples)

    input_data_numpy = np.array([whisper_prompt], dtype=object)
    input_data_numpy = input_data_numpy.reshape((1, 1))
    inputs[1].set_data_from_numpy(input_data_numpy)

    outputs = [protocol_client.InferRequestedOutput("TRANSCRIPTS")]

    sequence_id = np.random.randint(0, 1000000)
    response = triton_client.infer(
        model_name, inputs, request_id=str(sequence_id), outputs=outputs,
        headers={"Authorization": f"Basic {os.environ['BASIC_AUTH_TOKEN']}"}
    )

    decoding_results = response.as_numpy("TRANSCRIPTS")[0]
    if isinstance(decoding_results, np.ndarray):
        decoding_results = b" ".join(decoding_results).decode("utf-8")
    else:
        decoding_results = decoding_results.decode("utf-8")

    return decoding_results, duration


# Основная функция обработки аудиофайла
async def process_audio(update: Update, context):
    user = update.effective_user.username
    file = await context.bot.get_file(
        update.message.voice.file_id if update.message.voice else update.message.audio.file_id)

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_filename = temp_file.name
        await file.download_to_drive(temp_filename)

    logger.info(f"Получен аудиофайл от пользователя {user}, файл сохранён как {temp_filename}")

    # Конвертируем в WAV
    wav_file = convert_to_wav(temp_filename)
    logger.info(f"Файл конвертирован в WAV: {wav_file}")

    # Отправляем на Triton сервер
    triton_client = httpclient.InferenceServerClient(url=os.environ["INFERENCE_URL"], verbose=False)
    protocol_client = httpclient
    whisper_prompt = os.environ["WHISPER_PROMPT"] #"<|startoftranscript|><ru>"
    model_name = "whisper"

    try:
        transcript, duration = send_whisper(whisper_prompt, wav_file, model_name, triton_client, protocol_client)
        logger.info(f"Транскрипт: {transcript}")

        await update.message.reply_text(f"Распознанный текст:\n{transcript}")
    except Exception as e:
        logger.error(f"Ошибка при обработке: {e}")
        await update.message.reply_text(f"Произошла ошибка при обработке аудио:\n{e}")

    # Удаляем временные файлы
    os.remove(temp_filename)
    os.remove(wav_file)


# Команда старт
async def start(update: Update, context):
    await update.message.reply_text("Привет! Отправь аудиофайл или голосовое сообщение для распознавания.")


# Основная функция для запуска бота
def main():
    token = os.environ["TELEGRAM_BOT_TOKEN"]

    application = Application.builder().token(token).build()

    # Регистрация команд и обработчиков
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.AUDIO | filters.VOICE, process_audio))

    logger.info("Бот запущен и ожидает сообщений.")

    application.run_polling()


if __name__ == "__main__":
    main()
