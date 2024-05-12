from contextlib import asynccontextmanager
from typing import Dict, List, Tuple, Annotated
import os
import uvicorn
from fastapi import Body, FastAPI, File, UploadFile, status
from models.pipeline import Pipeline
from utils.data_models import PredictionResult


DEBUG = os.environ.get("DEBUG", False) in ("True", "true")


pipeline = dict()


@asynccontextmanager
async def lifespan(_: FastAPI):
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


def main() -> None:
    """Run application"""
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=DEBUG)


if __name__ == "__main__":
    main()
