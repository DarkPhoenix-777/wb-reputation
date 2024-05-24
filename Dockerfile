FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /app

COPY requirements.txt /app/

RUN pip install -r requirements.txt \
    && pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

VOLUME data

COPY . /app/

EXPOSE 8000

ENTRYPOINT [ "python", "main.py" ]