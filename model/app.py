from typing import Dict, List, Tuple
import requests
import numpy as np
import uvicorn
from fastapi import FastAPI, Body, File, UploadFile
from model import Model

app = FastAPI(debug=True)
model = Model()

MAX_IMAGES_IN_REQUEST = 2

@app.post("/prediction")
def get_prediction(files: List[UploadFile] = File(...)) -> str:
    contents = [f.file.read() for f in files]
    names = [f.filename for f in files]
    content_types = [f.content_type for f in files]

    files = []
    for i, name in enumerate(names):
        files.append(("files", (name, contents[i], content_types[i])))

    text_responses = []
    image_responses = []
    for i in range(0, len(files), MAX_IMAGES_IN_REQUEST):
        files_for_request = files[i:min(i+MAX_IMAGES_IN_REQUEST, len(files))]

        text_response = send_text_features_request(files_for_request)
        image_response = send_image_features_request(files_for_request)

        text_responses.append(text_response)
        image_responses.append(image_response)

    res = model.predict_on_imgs(names, text_responses, image_responses)
    return res

def send_image_features_request(files: List[Tuple[str, Tuple]]) -> requests.Response:
    image_response = requests.post("http://localhost:8002/image_encoder", files=files)
    return image_response

def send_text_features_request(files: List[Tuple[str, Tuple]]) -> requests.Response:
    texts = requests.post("http://localhost:8000/ocr", files=files)
    text_response = requests.post("http://localhost:8001/text_encoder", data=texts)
    return text_response

def main() -> None:
    """Run application"""
    uvicorn.run("app:app", host="localhost", port=8004, reload=True)


if __name__ == "__main__":
    main()
