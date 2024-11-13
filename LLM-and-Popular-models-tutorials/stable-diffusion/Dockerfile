FROM tritonserver:r24.01-diffusion
COPY deps ./deps
RUN pip install -r deps/requirements.txt && pip install deps/tritonserver-2.41.0.dev0-py3-none-any.whl python-multipart==0.0.9