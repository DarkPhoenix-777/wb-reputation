from typing import Dict, List
import os
import cv2
import json
import numpy as np
import uvicorn
from fastapi import FastAPI, Body, File, UploadFile, status
from model import OCR

DEBUG = os.environ.get("DEBUG", False) in ("True", "true")

app = FastAPI(debug=DEBUG)
OCR_model = OCR()


def read_image(image_bytes: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)


@app.post("/ocr")
def get_text(files: List[UploadFile] = File(...)) -> str:
    contents = [f.file.read() for f in files]
    names = [f.filename for f in files]
    content_types = [f.content_type for f in files]
    images = [read_image(image_bytes) for image_bytes in contents]
    texts = OCR_model.read_text(images)
    return texts


@app.get("/test")
def test():
    return status.HTTP_200_OK


def main() -> None:
    """Run application"""
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=DEBUG)


if __name__ == "__main__":
    main()
