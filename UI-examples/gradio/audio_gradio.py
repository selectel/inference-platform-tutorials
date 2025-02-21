import logging
import os
import tempfile
import time
from datetime import datetime

import gradio as gr
import numpy as np
import urllib.request
import tritonclient
import tritonclient.http as httpclient
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
        model_name, inputs, request_id=str(sequence_id), outputs=outputs, headers={"Authorization": "Basic YWRtaW46SWVjNWVuZ2E="}
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
    triton_client = httpclient.InferenceServerClient(url=server_url, verbose=False)
    protocol_client = httpclient

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

    return text, build_html_output(info)


title = "# Speech Recognition and Translation with Whisper"
description = """
This space shows how to do speech recognition and translation with Nvidia **Triton**.

Please visit
<https://huggingface.co/yuekai/model_repo_whisper_large_v2>
for triton speech recognition.

The service is running on a GPU based on triton server.

See more information by visiting the following links:

- <https://github.com/triton-inference-server>
- <https://github.com/yuekaizhang/Triton-ASR-Client/tree/main>
- <https://github.com/k2-fsa/sherpa/tree/master/triton>
- <https://github.com/wenet-e2e/wenet/tree/main/runtime/gpu>
- <https://github.com/alibaba-damo-academy/FunASR/tree/main/funasr/runtime/triton_gpu>

"""

# css style is copied from
# https://huggingface.co/spaces/alphacep/asr/blob/main/app.py#L113
css = """
.result {display:flex;flex-direction:column}
.result_item {padding:15px;margin-bottom:8px;border-radius:15px;width:100%}
.result_item_success {background-color:mediumaquamarine;color:white;align-self:start}
.result_item_error {background-color:#ff7070;color:white;align-self:start}
"""


# def update_model_dropdown(language: str):
#     if language in language_to_models:
#         choices = language_to_models[language]
#         return gr.Dropdown.update(choices=choices, value=choices[0])

#     raise ValueError(f"Unsupported language: {language}")


demo = gr.Blocks(css=css)


with demo:
    gr.Markdown(title)
    language_choices = ["Chinese", "English", "Chinese+English", "Korean", "Japanese", "Arabic", "German", "French", "Russian"]
    server_url_textbox = gr.Textbox(
        label='Triton Inference Server URL',
        value='10.19.203.82:8001',
        placeholder='e.g. localhost:8001',
        max_lines=1,
    )

    whisper_prompt_textbox = gr.Textbox(
        label='Whisper prompt',
        placeholder='Whisper prompt e.g. <|startoftranscript|><zh><en><transcribe>',
        max_lines=1,
    )
    language_radio = gr.Radio(
        label="Language",
        choices=language_choices,
        value=language_choices[0],
    )
    model_dropdown = gr.Dropdown(
        choices=["whisper-large-v2"],
        label="Select a model",
        value="whisper-large-v2",
    )

    # language_radio.change(
    #     update_model_dropdown,
    #     inputs=language_radio,
    #     outputs=model_dropdown,
    # )

    decoding_method_radio = gr.Radio(
        label="Decoding method",
        choices=["greedy_search"],
        value="greedy_search",
    )

    # whisper_prompt_textbox_slider = gr.Slider(
    #     minimum=1,
    #     value=4,
    #     step=1,
    #     label="Number of active paths for modified_beam_search",
    # )

    with gr.Tabs():
        with gr.TabItem("Upload from disk"):
            uploaded_file = gr.Audio(
                sources=["upload"],
                type="filepath",
                label="Upload from disk",
            )
            upload_button = gr.Button("Submit for recognition")
            uploaded_output = gr.Textbox(label="Recognized speech from uploaded file")
            uploaded_html_info = gr.HTML(label="Info")

            # gr.Examples(
            #     examples=examples,
            #     inputs=[
            #         language_radio,
            #         model_dropdown,
            #         decoding_method_radio,
            #         whisper_prompt_textbox,
            #         uploaded_file,
            #     ],
            #     outputs=[uploaded_output, uploaded_html_info],
            #     fn=process_uploaded_file,
            #     cache_examples=False,
            # )

        with gr.TabItem("Record from microphone"):
            microphone = gr.Audio(
                sources=["microphone"],  # Choose between "microphone", "upload"
                type="filepath",
                label="Record from microphone",
            )

            record_button = gr.Button("Submit for recognition")
            recorded_output = gr.Textbox(label="Recognized speech from recordings")
            recorded_html_info = gr.HTML(label="Info")

            # gr.Examples(
            #     examples=examples,
            #     inputs=[
            #         language_radio,
            #         model_dropdown,
            #         decoding_method_radio,
            #         whisper_prompt_textbox,
            #         microphone,
            #     ],
            #     outputs=[recorded_output, recorded_html_info],
            #     fn=process_microphone,
            #     cache_examples=False,
            # )

        with gr.TabItem("From URL"):
            url_textbox = gr.Textbox(
                    max_lines=1,
                    placeholder="URL to an audio file",
                    label="URL",
                    interactive=True,
            )

            url_button = gr.Button("Submit for recognition")
            url_output = gr.Textbox(label="Recognized speech from URL")
            url_html_info = gr.HTML(label="Info")

        upload_button.click(
            process_uploaded_file,
            inputs=[
                language_radio,
                model_dropdown,
                decoding_method_radio,
                whisper_prompt_textbox,
                uploaded_file,
                server_url_textbox,
            ],
            outputs=[uploaded_output, uploaded_html_info],
        )

        record_button.click(
            process_microphone,
            inputs=[
                language_radio,
                model_dropdown,
                decoding_method_radio,
                whisper_prompt_textbox,
                microphone,
                server_url_textbox,
            ],
            outputs=[recorded_output, recorded_html_info],
        )

        url_button.click(
            process_url,
            inputs=[
                language_radio,
                model_dropdown,
                decoding_method_radio,
                whisper_prompt_textbox,
                url_textbox,
                server_url_textbox,
            ],
            outputs=[url_output, url_html_info],
        )

    gr.Markdown(description)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)

    demo.launch(share=False, server_name='0.0.0.0')