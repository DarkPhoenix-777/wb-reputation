from typing import Dict, List
import json
import uvicorn
from fastapi import FastAPI, Body
from text_encoder import Text_encoder

app = FastAPI(debug=True)
feature_extractor = Text_encoder()


@app.post("/text_encoder")
def get_features(texts: List[str] = Body(...)) -> str:
    features = feature_extractor.get_text_embeddings(texts)
    return json.dumps(features.tolist())

@app.get("/test")
def test():
    return "ok"


def main() -> None:
    """Run application"""
    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=True)


if __name__ == "__main__":
    main()
