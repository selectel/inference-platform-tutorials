FROM nvcr.io/nvidia/tritonserver:24.07-trtllm-python-py3
LABEL maintainer="NVIDIA"
LABEL repository="tritonserver"

COPY deps ./deps
RUN pip install -r deps/requirements.txt && pip install deps/tritonserver-2.41.0.dev0-py3-none-any.whl

RUN apt update && apt-get install -y ffmpeg
RUN python3 -m pip install mpmath==1.3.0 gradio==3.50.2 tritonclient[all]==2.49.0 python-multipart==0.0.9

COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt
RUN git clone https://github.com/NVIDIA/TensorRT-LLM.git
