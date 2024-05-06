from typing import Dict, List
import numpy as np
import requests
import json

class FeatureExtractor():
    """get image + text embeddings"""
    def __init__(self) -> None:
        pass


    def get_features(self, images, image_names, content_types):
        """
        Get features from model

        Parameters
        ----------

        Returns
        -------
        np.ndarray
            2d array of concatenated images an text embeddings
        """

        files = []
        for i, image in enumerate(images):
            files.append(("files", (image_names[i], image, content_types[i])))

        text_response = requests.post("http://localhost:8001/text_encoder", files=files)
        image_response = requests.post("http://localhost:8002/image_encoder", files=files)
        
        text_features = np.asarray(json.loads(text_response.json()))
        image_features = np.asarray(json.loads(image_response.json()))


        return np.concatenate([text_features, image_features], axis=1)
