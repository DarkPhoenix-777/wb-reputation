import os
import json
import numpy as np
import onnxruntime
import torch


class Classifier():
    """classifier"""
    def __init__(self, model_file: str="models/models_onnx/classifier.onnx",
                 providers=None) -> None:

        self.onnx_session = onnxruntime.InferenceSession(model_file,
                                                         providers=providers)


    def predict(self, features: np.ndarray, batch_size=512) -> np.ndarray:
        """
        Predict on features

        Parameters
        ----------
        features : np.ndarray
            2d array of concatenated images an text embeddings, shape: (n_images, 1280)
        batch_size : int
            batch size for classifier 

        Returns
        -------
        np.ndarray
            array of probabilities
        """
        probabilities = []

        for i in range(0, len(features), batch_size):
            batch = features[i:min(i+batch_size, len(features))]
            probabilities.extend(self.onnx_session.run(None, {'input': batch})[0])

        return np.concatenate(probabilities)
