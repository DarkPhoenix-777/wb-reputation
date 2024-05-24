from pydantic import BaseModel

class PredictionResult(BaseModel):
    image: str | int
    prob: float
    verdict: bool