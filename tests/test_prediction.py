import io
import os
from PIL import Image
import numpy as np
from models.pipeline import Pipeline
from utils.data_models import PredictionResult


def image_to_byte_array(image: Image) -> bytes:
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format=image.format)
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr


def test_single_image() -> None:
    """Test prediction for single image"""
    pipeline = Pipeline()
    img_name = "59359031.jpg"
    img = [image_to_byte_array(Image.open("tests/test_images/" + img_name))]
    pred = pipeline.predict_on_bytes(img)

    assert isinstance(pred[0], PredictionResult), "Result has incorrect type"
    assert not pred[0].verdict, "Result has incorrect prediction"


def test_multiple_images() -> None:
    """Test prediction for multiple images"""
    pipeline = Pipeline()
    img_names = ["59359031.jpg", "123152360.jpg"]
    imgs = [image_to_byte_array(Image.open("tests/test_images/" + img_name)) for img_name in img_names]
    pred = pipeline.predict_on_bytes(imgs)

    assert isinstance(pred[0],PredictionResult), "Result has incorrect type"
    assert isinstance(pred[1],PredictionResult), "Result has incorrect type"
    assert not pred[0].verdict, "Result has incorrect prediction"
    assert pred[1].verdict, "Result has incorrect prediction"
