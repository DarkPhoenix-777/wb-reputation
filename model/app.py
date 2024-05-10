from typing import Dict, List, Tuple
import os
import asyncio
import httpx
import numpy as np
import uvicorn
from fastapi import FastAPI, Body, File, UploadFile, status
from model import Model

DEBUG = os.environ.get("DEBUG", False) in ("True", "true")

app = FastAPI(debug=DEBUG)
model = Model()

MAX_IMAGES_IN_REQUEST = int(os.environ.get("MAX_IMAGES_IN_REQUEST", -1))

if DEBUG:
    ocr_host = "localhost"
    text_encoder_host = "localhost"
    image_encoder_host = "localhost"
else:
    ocr_host = "ocr"
    text_encoder_host = "text_encoder"
    image_encoder_host = "image_encoder"


@app.post("/prediction")
async def get_prediction(files: List[UploadFile] = File(...)) -> str:
    contents = [f.file.read() for f in files]
    names = [f.filename for f in files]
    content_types = [f.content_type for f in files]

    files = []
    for i, name in enumerate(names):
        files.append(("files", (name, contents[i], content_types[i])))

    text_responses = []
    image_responses = []

    if MAX_IMAGES_IN_REQUEST != -1:
        for i in range(0, len(files), MAX_IMAGES_IN_REQUEST):
            files_for_request = files[i:min(i+MAX_IMAGES_IN_REQUEST, len(files))]

            text_response_task = send_text_features_request(files_for_request)
            image_response_task = send_image_features_request(files_for_request)

            text_responses.append(await text_response_task)
            image_responses.append(await image_response_task)

    else:
        text_response_task = send_text_features_request(files)
        image_response_task = send_image_features_request(files)

        text_responses.append(await text_response_task)
        image_responses.append(await image_response_task)

    res = model.predict_on_imgs(names, text_responses, image_responses)
    return res


async def send_image_features_request(files: List[Tuple[str, Tuple]]) -> httpx.Response:
    async with httpx.AsyncClient() as client:
        image_response = await client.post(f"http://{image_encoder_host}:8002/image_encoder", files=files)
    return image_response


async def send_text_features_request(files: List[Tuple[str, Tuple]]) -> httpx.Response:
    async with httpx.AsyncClient() as client:
        texts = await client.post(f"http://{ocr_host}:8000/ocr", files=files)
        text_response = await client.post(f"http://{text_encoder_host}:8001/text_encoder", data=texts.text)
    return text_response


async def check_url(url: str) -> None:
    async with httpx.AsyncClient() as client:
        try:
            r = await client.get(url)
            r.raise_for_status()
        except Exception as e:
            print(f"Error fetching {url}: {e}")


@app.get("/test")
async def test():
    return status.HTTP_200_OK


@app.on_event("startup")
async def check_connections() -> None:
    """Проверка соединения с другими сервисами"""
    tasks = [
        check_url(f"http://{ocr_host}:8000/test"),
        check_url(f"http://{text_encoder_host}:8001/test"),
        check_url(f"http://{image_encoder_host}:8002/test")
    ]
    await asyncio.gather(*tasks)


def main() -> None:
    """Run application"""
    uvicorn.run("app:app", host="0.0.0.0", port=8004, reload=DEBUG)


if __name__ == "__main__":
    main()
