from typing import Dict, List
import os
import json
import uvicorn
from fastapi import FastAPI, Body, status
from text_encoder import Text_encoder

DEBUG = os.environ.get("DEBUG", False) in ("True", "true")

app = FastAPI(debug=DEBUG)
text_encoder = Text_encoder()


@app.post("/text_encoder")
def get_features(texts: List[str] = Body(...)) -> str:
    features = text_encoder.get_text_embeddings(texts)
    return json.dumps(features.tolist())


@app.get("/test")
def test():
    return status.HTTP_200_OK


def main() -> None:
    """Run application"""
    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=DEBUG)


if __name__ == "__main__":
    main()
