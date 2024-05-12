import io
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from PIL import Image

from models.ocr import OCR
from models.text_encoder import TextEncoder
from models.image_encoder import ImageEncoder
from models.classifier import Classifier
from utils.data_models import PredictionResult
from utils.download_models import download_model


THRESHOLD = 0.5


class Pipeline():
    """pipeline"""
    def __init__(self) -> None:
        if torch.cuda.is_available():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
            print("Warning CUDA not detected by torch")

        # Проверка на наличие моделей + скачивание
        if not os.path.isfile("models_onnx/distilbert-base-uncased.onnx"):
            print("Text encoder not found")
            download_model("distilbert-base-uncased")
        if not os.path.isfile("models_onnx/clip_image_encoder.onnx"):
            print("Image encoder not found")
            download_model("clip_image_encoder")
        if not os.path.isfile("models_onnx/classifier.onnx"):
            print("Classifier not found")
            download_model("classifier")

        self.ocr = OCR()
        self.text_encoder = TextEncoder(providers=providers)
        self.image_encoder = ImageEncoder(providers=providers)
        self.classifier = Classifier(providers=providers)

    @staticmethod
    def read_image(image_bytes: bytes) -> np.ndarray:
        """Load image from bytes to PIL Image"""
        image = Image.open(io.BytesIO(image_bytes))
        return image


    def predict_on_bytes(self, contents: List[bytes],
                         image_names: List[str] | None = None) -> List[PredictionResult]:
        """
        Predict on image bytes

        Parameters
        ----------
        contents: List[bytes]
            images in bytes
        image_names: List[str]
            if None, sets number of image
        
        Returns
        -------
        List[PredictionResult]
            predictions
        """
        if image_names is None:
            image_names = range(len(contents))

        contents = [self.read_image(content) for content in contents]

        texts = self.ocr.read_text(contents)
        texts_features = self.text_encoder.get_text_embeddings(texts)
        image_features = self.image_encoder.get_img_embeddings(contents)

        features = self.get_features(texts_features, image_features).astype(np.float32)

        probabilities = self.classifier.predict(features).tolist()

        res = []
        for i, name in enumerate(image_names):
            res.append(PredictionResult(image=name,
                                        prob=probabilities[i],
                                        verdict=probabilities[i] >= THRESHOLD))
        return res



    def get_features(self, text_features: np.ndarray, image_features: np.ndarray) -> np.ndarray:
        """
        Get features for model + shape checks

        Parameters
        ----------
        text_features: np.ndarray
            features from text_encoder, shape: (n_images, 768)
        image_features: np.ndarray
            features from image encoder, shape: (n_images, 512)

        Returns
        -------
        np.ndarray
            2d array of concatenated images an text embeddings, shape: (n_images, 1280)
        """

        assert text_features.shape[1] == 768, \
            ValueError("Text features has incorrect shape on axis 1")
        assert image_features.shape[1] == 512, \
            ValueError("Image features has incorrect shape on axis 1")
        assert text_features.shape[0] == image_features.shape[0], \
            ValueError("Image and text features has different length")

        return np.concatenate([text_features, image_features], axis=1)
