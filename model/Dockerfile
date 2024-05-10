FROM pytorch/pytorch:2.3.0-cuda11.8-cudnn8-runtime

WORKDIR /app

COPY requirements.txt /app/

RUN pip install -r requirements.txt \
    && pip install onnxruntime-gpu

COPY . /app/

EXPOSE 8004

ENTRYPOINT [ "python", "app.py" ]