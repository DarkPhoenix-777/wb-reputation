from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Tuple, Annotated
import os
import uvicorn
from fastapi import Body, FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse

from models.pipeline import Pipeline
from utils.train import Trainer
from utils.data_models import PredictionResult
from utils.download_models import download_model

DEBUG = os.environ.get("DEBUG", False) in ("True", "true")
ALLOW_TRAIN = os.environ.get("ALLOW_TRAIN", False) in ("True", "true")

pipelines = dict()


@asynccontextmanager
async def lifespan(_: FastAPI):
    pipelines["main_pipeline"] = Pipeline()
    yield
    pipelines.clear()


app = FastAPI(debug=DEBUG, lifespan=lifespan)


@app.post("/prediction")
async def get_prediction(file: bytes = Body(...)) -> PredictionResult:
    """Предсказания по одному файлу"""
    content = [file]
    name = [0]
    return pipelines["main_pipeline"].predict_on_bytes(content, name)[0]


@app.post("/prediction_batch")
async def get_prediction_batch(files: Annotated[List[UploadFile], File()]) -> List[PredictionResult]:
    """Предсказания по нескольким файлам"""
    contents = [f.file.read() for f in files]
    names = [f.filename for f in files]
    return pipelines["main_pipeline"].predict_on_bytes(contents, names)


@app.get("/train")
async def train(n_epoch: Optional[int] = 10,
                early_stopping_rounds: Optional[int] = None,
                learning_rate: Optional[float] = 1e-3,
                log_batch: Optional[bool] = True):
    """
    Дообучение модели

    Parameters
    ----------
    n_epoch: int
        Максимальное количество эпох
    early_stopping_rounds: Optional[int]
        Если None, то будет обучаться заданное количество эпох
        Если int, то остановит обучение, если лосс на валидации не уменьшается заданное количество эпох
    log_batch: bool
        Если True - выводит лосс на обучении по батчу в консоль

    Returns
    -------
    status
        http статус
    """
    if not ALLOW_TRAIN:
        raise HTTPException(
            status_code=status.HTTP_423_LOCKED,
            detail = "Training disabled"
        )
    else:
        trainer = Trainer(pipelines["main_pipeline"], n_epoch, early_stopping_rounds, learning_rate, log_batch)
        trainer.train()
        trainer.save_classifier("models/models_onnx/")
        return status.HTTP_200_OK


@app.get("/download_classifier")
async def download_classifier():
    """Скачивание классификатора"""
    if os.path.isfile("models/models_onnx/classifier.onnx"):
        return FileResponse("models/models_onnx/classifier.onnx", 
                            media_type='application/octet-stream', 
                            filename="classifier.onnx")
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail = "Classifier.onnx not found"
        )


@app.get("/test")
async def test():
    """For connection check"""
    return status.HTTP_200_OK


def main() -> None:
    """Run application"""

    # Проверка на наличие моделей + скачивание
    if not os.path.isfile("models/models_onnx/distilbert-base-uncased.onnx"):
        print("Text encoder not found")
        download_model("distilbert-base-uncased")
    if not os.path.isfile("models/models_onnx/clip_image_encoder.onnx"):
        print("Image encoder not found")
        download_model("clip_image_encoder")
    if not os.path.isfile("models/models_onnx/classifier.onnx"):
        print("Classifier not found")
        download_model("classifier")


    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=DEBUG)


if __name__ == "__main__":
    main()
