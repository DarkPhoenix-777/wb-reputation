from typing import Dict, List
import cv2
import os
import json
import numpy as np
import uvicorn
from fastapi import FastAPI, Body, File, UploadFile, status
from image_encoder import Image_encoder

DEBUG = os.environ.get("DEBUG", False) in ("True", "true")

app = FastAPI(debug=DEBUG)
image_encoder = Image_encoder()


@app.post("/image_encoder")
def get_features(files: List[UploadFile] = File(...)) -> str:
    contents = [f.file.read() for f in files]
    names = [f.filename for f in files]
    content_types = [f.content_type for f in files]
    images = [cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR) for image_bytes in contents]
    features = image_encoder.get_img_embeddings(images)
    return json.dumps(features.tolist())


@app.get("/test")
def test():
    return status.HTTP_200_OK


def main() -> None:
    """Run application"""
    uvicorn.run("app:app", host="0.0.0.0", port=8002, reload=DEBUG)


if __name__ == "__main__":
    main()
