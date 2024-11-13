import logging
import os
import tempfile
import time
from datetime import datetime

import numpy as np
import urllib.request
import tritonclient
import tritonclient.grpc as grpcclient
from tritonclient.utils import np_to_triton_dtype
import soundfile


def convert_to_wav(in_filename: str) -> str:
    """Convert the input audio file to a wave file"""
    out_filename = in_filename + ".wav"
    if '.mp3' in in_filename:
        _ = os.system(f"ffmpeg -y -i '{in_filename}' -acodec pcm_s16le -ac 1 -ar 16000 '{out_filename}' || exit 1")
    else:
        _ = os.system(f"ffmpeg -hide_banner -y -i '{in_filename}' -ar 16000 '{out_filename}' || exit 1")
    return out_filename


def build_html_output(s: str, style: str = "result_item_success"):
    return f"""
    <div class='result'>
        <div class='result_item {style}'>
          {s}
        </div>
    </div>
    """

def process_url(
    language: str,
    repo_id: str,
    decoding_method: str,
    whisper_prompt_textbox: str,
    url: str,
    server_url_textbox: str,
):
    logging.info(f"Processing URL: {url}")
    with tempfile.NamedTemporaryFile() as f:
        try:
            urllib.request.urlretrieve(url, f.name)

            return process(
                in_filename=f.name,
                language=language,
                repo_id=repo_id,
                decoding_method=decoding_method,
                whisper_prompt_textbox=whisper_prompt_textbox,
                server_url=server_url_textbox,
            )
        except Exception as e:
            logging.info(str(e))
            return "", build_html_output(str(e), "result_item_error")

def process_uploaded_file(
    language: str,
    repo_id: str,
    decoding_method: str,
    whisper_prompt_textbox: int,
    in_filename: str,
    server_url_textbox: str,
):
    if in_filename is None or in_filename == "":
        return "", build_html_output(
            "Please first upload a file and then click "
            'the button "submit for recognition"',
            "result_item_error",
        )

    logging.info(f"Processing uploaded file: {in_filename}")
    try:
        return process(
            in_filename=in_filename,
            language=language,
            repo_id=repo_id,
            decoding_method=decoding_method,
            whisper_prompt_textbox=whisper_prompt_textbox,
            server_url=server_url_textbox,
        )
    except Exception as e:
        logging.info(str(e))
        return "", build_html_output(str(e), "result_item_error")


def process_microphone(
    language: str,
    repo_id: str,
    decoding_method: str,
    whisper_prompt_textbox: str,
    in_filename: str,
    server_url_textbox: str,
):
    if in_filename is None or in_filename == "":
        return "", build_html_output(
            "Please first click 'Record from microphone', speak, "
            "click 'Stop recording', and then "
            "click the button 'submit for recognition'",
            "result_item_error",
        )

    logging.info(f"Processing microphone: {in_filename}")
    try:
        return process(
            in_filename=in_filename,
            language=language,
            repo_id=repo_id,
            decoding_method=decoding_method,
            whisper_prompt_textbox=whisper_prompt_textbox,
            server_url=server_url_textbox,
        )
    except Exception as e:
        logging.info(str(e))
        return "", build_html_output(str(e), "result_item_error")

def send_whisper(whisper_prompt, wav_path, model_name, triton_client, protocol_client, padding_duration=10):
    waveform, sample_rate = soundfile.read(wav_path)
    assert sample_rate == 16000, f"Only support 16k sample rate, but got {sample_rate}"
    duration = int(len(waveform) / sample_rate)

    # padding to nearset 10 seconds
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
    # generate a random sequence id
    sequence_id = np.random.randint(0, 1000000)

    response = triton_client.infer(
        model_name, inputs, request_id=str(sequence_id), outputs=outputs
    )

    decoding_results = response.as_numpy("TRANSCRIPTS")[0]
    if type(decoding_results) == np.ndarray:
        decoding_results = b" ".join(decoding_results).decode("utf-8")
    else:
        # For wenet
        decoding_results = decoding_results.decode("utf-8")
    return decoding_results, duration

def process(
    language: str,
    repo_id: str,
    decoding_method: str,
    whisper_prompt_textbox: str,
    in_filename: str,
    server_url: str,
):
    logging.info(f"language: {language}")
    logging.info(f"repo_id: {repo_id}")
    logging.info(f"decoding_method: {decoding_method}")
    logging.info(f"whisper_prompt_textbox: {whisper_prompt_textbox}")
    logging.info(f"in_filename: {in_filename}")

    model_name = "whisper"
    triton_client = grpcclient.InferenceServerClient(url=server_url, verbose=False)
    protocol_client = grpcclient

    filename = convert_to_wav(in_filename)

    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d %H:%M:%S.%f")
    logging.info(f"Started at {date_time}")

    start = time.time()

    text, duration = send_whisper(whisper_prompt_textbox, filename, model_name, triton_client, protocol_client)

    date_time = now.strftime("%Y-%m-%d %H:%M:%S.%f")
    end = time.time()

    #metadata = torchaudio.info(filename)
    #duration = metadata.num_frames / sample_rate
    rtf = (end - start) / duration

    logging.info(f"Finished at {date_time} s. Elapsed: {end - start: .3f} s")

    info = f"""
    Wave duration  : {duration: .3f} s <br/>
    Processing time: {end - start: .3f} s <br/>
    RTF: {end - start: .3f}/{duration: .3f} = {rtf:.3f} <br/>
    """
    if rtf > 1:
        info += (
            "<br/>We are loading the model for the first run. "
            "Please run again to measure the real RTF.<br/>"
        )

    logging.info(info)
    logging.info(f"\nrepo_id: {repo_id}\nhyp: {text}")

    return text


if __name__ == "__main__":
    in_filename = "<file in m4a, mp3, wav formats>"
    language = "English"
    repo_id = "whisper-large-v2"
    decoding_method = "greedy_search"
    whisper_prompt_textbox = ""
    server_url_textbox = "localhost:8001" # url до инференс сервера

    text = process(
        in_filename=in_filename,
        language=language,
        repo_id=repo_id,
        decoding_method=decoding_method,
        whisper_prompt_textbox=whisper_prompt_textbox,
        server_url=server_url_textbox,
    )
    print(text)