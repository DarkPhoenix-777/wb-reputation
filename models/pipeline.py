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


THRESHOLD = 0.5
OCR_BATCH_SIZE = int(os.environ.get("OCR_BATCH_SIZE", 64))
TEXT_ENCODER_BATCH_SIZE = int(os.environ.get("TEXT_ENCODER_BATCH_SIZE", 64))
IMAGE_ENCODER_BATCH_SIZE = int(os.environ.get("IMAGE_ENCODER_BATCH_SIZE", 64))
CLASSIFIER_BATCH_SIZE = int(os.environ.get("CLASSIFIER_BATCH_SIZE", 512))

class Pipeline():
    """pipeline"""
    def __init__(self) -> None:
        if torch.cuda.is_available():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
            print("Warning CUDA not detected by torch")

        self.ocr = OCR(batch_size=OCR_BATCH_SIZE)
        self.text_encoder = TextEncoder(providers=providers, batch_size=TEXT_ENCODER_BATCH_SIZE)
        self.image_encoder = ImageEncoder(providers=providers, batch_size=IMAGE_ENCODER_BATCH_SIZE)
        self.classifier = Classifier(providers=providers, batch_size=CLASSIFIER_BATCH_SIZE)

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

        features = self.get_features_bytes(contents)

        probabilities = self.classifier.predict(features).tolist()

        res = []
        for i, name in enumerate(image_names):
            res.append(PredictionResult(image=name,
                                        prob=probabilities[i],
                                        verdict=probabilities[i] >= THRESHOLD))
        return res


    def get_features_bytes(self, contents: List[bytes]) -> np.ndarray:
        """
        Get features for model from list of images + shape checks

        Parameters
        ----------
        contents: List[bytes]
            images in bytes

        Returns
        -------
        np.ndarray
            2d array of concatenated images an text embeddings, shape: (n_images, 1280)
        """
        
        contents = [self.read_image(content) for content in contents]
        return self.get_features_from_images(contents)


    def get_features_from_images(self, contents: List[np.ndarray]) -> np.ndarray:
        """
        Get features for model from list of images + shape checks

        Parameters
        ----------
        contents: List[np.ndarray]
            images

        Returns
        -------
        np.ndarray
            2d array of concatenated images an text embeddings, shape: (n_images, 1280)
        """

        texts = self.ocr.read_text(contents)
        texts_features = self.text_encoder.get_text_embeddings(texts)
        image_features = self.image_encoder.get_img_embeddings(contents)

        assert texts_features.shape[1] == 768, \
            ValueError("Text features has incorrect shape on axis 1")
        assert image_features.shape[1] == 512, \
            ValueError("Image features has incorrect shape on axis 1")
        assert texts_features.shape[0] == image_features.shape[0], \
            ValueError("Image and text features has different length")

        return np.concatenate([texts_features, image_features], axis=1).astype(np.float32)
