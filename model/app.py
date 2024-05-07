from typing import Dict, List
import requests
import numpy as np
import uvicorn
from fastapi import FastAPI, Body, File, UploadFile
from model import Model

app = FastAPI(debug=True)
model = Model()

@app.post("/prediction")
def get_prediction(files: List[UploadFile] = File(...)) -> str:
    contents = [f.file.read() for f in files]
    names = [f.filename for f in files]
    content_types = [f.content_type for f in files]

    files = []
    for i, name in enumerate(names):
        files.append(("files", (name, contents[i], content_types[i])))

    texts = requests.post("http://localhost:8000/ocr", files=files)
    text_response = requests.post("http://localhost:8001/text_encoder", data=texts)
    image_response = requests.post("http://localhost:8002/image_encoder", files=files)

    res = model.predict_on_imgs(names, text_response, image_response)
    return res


def main() -> None:
    """Run application"""
    uvicorn.run("app:app", host="localhost", port=8004, reload=True)


if __name__ == "__main__":
    main()
