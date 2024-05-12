from contextlib import asynccontextmanager
from typing import Dict, List, Tuple, Annotated
import os
import uvicorn
from fastapi import Body, FastAPI, File, UploadFile, status
from models.pipeline import Pipeline
from utils.data_models import PredictionResult
from utils.download_models import download_model


DEBUG = os.environ.get("DEBUG", False) in ("True", "true")


pipeline = dict()


@asynccontextmanager
async def lifespan(_: FastAPI):
    check_models()
    pipeline["main_pipline"] = Pipeline()
    yield
    pipeline.clear()


app = FastAPI(debug=DEBUG, lifespan=lifespan)


@app.post("/prediction")
async def get_prediction(file: bytes = Body(...)) -> PredictionResult:
    content = [file]
    name = [0]
    return pipeline["main_pipline"].predict_on_bytes(content, name)[0]


@app.post("/prediction_batch")
async def get_prediction_batch(files: Annotated[List[UploadFile], File()]) -> List[PredictionResult]:
    contents = [f.file.read() for f in files]
    names = [f.filename for f in files]
    return pipeline["main_pipline"].predict_on_bytes(contents, names)


@app.get("/test")
async def test():
    """For connection check"""
    return status.HTTP_200_OK


def check_models() -> None:
    """Проверка на наличие моделей + скачивание"""
    if not os.path.isfile("models/models_onnx/distilbert-base-uncased.onnx"):
        print("Text encoder not found. Downloading")
        download_model("distilbert-base-uncased")
    if not os.path.isfile("models/models_onnx/clip_image_encoder.onnx"):
        print("Image encoder not found. Downloading")
        download_model("clip_image_encoder")
    if not os.path.isfile("models/models_onnx/classifier.onnx"):
        print("Classifier not found. Downloading")
        download_model("classifier")


def main() -> None:
    """Run application"""
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=DEBUG)


if __name__ == "__main__":
    main()
