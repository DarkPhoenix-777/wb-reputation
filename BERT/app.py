from typing import Dict, List
import json
import numpy as np
import requests
import uvicorn
from fastapi import FastAPI, Body
from text_encoder import Text_encoder

app = FastAPI(debug=True)
feature_extractor = Text_encoder()


@app.post("/text_encoder")
def get_features(image_bytes: bytes = Body(bytes)) -> str:
    headers = {'Content-Type': 'application/octet-stream'}
    texts = requests.post("http://localhost:8000/ocr", headers=headers, data=image_bytes).text
    features = feature_extractor.get_text_embeddings([texts])
    return json.dumps(features.tolist())


def main() -> None:
    """Run application"""
    uvicorn.run("app:app", host="localhost", port=8001, reload=True)


if __name__ == "__main__":
    main()
