from typing import Dict, List
import cv2
import json
import numpy as np
import uvicorn
from fastapi import FastAPI, Body
from model import OCR

app = FastAPI(debug=True)
OCR_model = OCR()


# @app.post("/ocr")
# def get_text(image: np.ndarray) -> str:
#     features = OCR_model.read_text(image)
#     return json.dumps(features.tolist())

@app.post("/ocr")
def get_text(image_bytes: bytes = Body(bytes)) -> str:
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    text = OCR_model.read_text(image)
    return text


def main() -> None:
    """Run application"""
    uvicorn.run("app:app", host="localhost", port=8000, reload=True)


if __name__ == "__main__":
    main()
