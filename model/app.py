from typing import Dict, List
import random
import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, Body, File, UploadFile
from model import Model

app = FastAPI(debug=True)
model = Model()

@app.get("/prediction/random")
def get_random_prediction(data: str) -> Dict:
    """get predictions"""
    score = random.random()
    pred = {
        "score": score,
        "target": score >= 0.5
    }
    return pred


@app.post("/prediction")
def get_prediction(files: List[UploadFile] = File(...)) -> str:
    contents = [f.file.read() for f in files]
    names = [f.filename for f in files]
    content_types = [f.content_type for f in files]
    res = model.predict_on_imgs(contents, names, content_types)
    return res


def main() -> None:
    """Run application"""
    uvicorn.run("app:app", host="localhost", port=8004, reload=True)


if __name__ == "__main__":
    main()
