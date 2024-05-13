import os
from typing import Dict, List
import json
import numpy as np
import onnxruntime
import torch


class Classifier():
    """classifier"""
    def __init__(self, model_file: str="models/models_onnx/classifier.onnx",
                 providers: List[str] = None,
                 batch_size: int = 512) -> None:

        self.batch_size = batch_size if batch_size > 0 else 512
        self.onnx_session = onnxruntime.InferenceSession(model_file,
                                                         providers=providers)


    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict on features

        Parameters
        ----------
        features : np.ndarray
            2d array of concatenated images an text embeddings, shape: (n_images, 1280)

        Returns
        -------
        np.ndarray
            array of probabilities
        """
        probabilities = []

        for i in range(0, len(features), self.batch_size):
            batch = features[i:min(i+self.batch_size, len(features))]
            probabilities.extend(self.onnx_session.run(None, {'input': batch})[0])

        return np.concatenate(probabilities)
