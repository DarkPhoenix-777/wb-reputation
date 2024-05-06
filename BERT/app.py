from typing import Dict, List
import json
import numpy as np
import requests
import uvicorn
from fastapi import FastAPI, Body, File, UploadFile
from text_encoder import Text_encoder

app = FastAPI(debug=True)
feature_extractor = Text_encoder()


@app.post("/text_encoder")
def get_features(files: List[UploadFile] = File(...)) -> str:
    contents = [f.file.read() for f in files]
    names = [f.filename for f in files]
    content_types = [f.content_type for f in files]
    headers = {'Content-Type': 'application/octet-stream'}
    texts = [requests.post("http://localhost:8000/ocr", headers=headers, data=image_bytes).text for image_bytes in contents]
    features = feature_extractor.get_text_embeddings(texts)
    return json.dumps(features.tolist())


def main() -> None:
    """Run application"""
    uvicorn.run("app:app", host="localhost", port=8001, reload=True)


if __name__ == "__main__":
    main()
