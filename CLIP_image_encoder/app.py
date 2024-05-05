from typing import Dict, List
import cv2
import json
import numpy as np
import uvicorn
from fastapi import FastAPI, Body
from image_encoder import Image_encoder

app = FastAPI(debug=True)
image_encoder = Image_encoder()


@app.post("/image_encoder")
def get_features(image_bytes: bytes = Body(bytes)) -> str:
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    features = image_encoder.get_img_embeddings([image])
    return json.dumps(features.tolist())


def main() -> None:
    """Run application"""
    uvicorn.run("app:app", host="localhost", port=8002, reload=True)


if __name__ == "__main__":
    main()
