from typing import Dict, List
import numpy as np
import requests
import json

class FeatureExtractor():
    """get image + text embeddings"""
    def __init__(self) -> None:
        pass


    def get_features(self, image_bytes_arr):
        """
        Get features from model

        Parameters
        ----------
        imgs_list : List[np.ndarray]
            list of images in RGB

        Returns
        -------
        np.ndarray
            2d array of concatenated images an text embeddings
        """
        text_features = []
        image_features = []
        headers = {'Content-Type': 'application/octet-stream'}

        for image_bytes in image_bytes_arr:

            text_response = requests.post("http://localhost:8001/text_encoder", data=image_bytes, headers=headers)
            image_response = requests.post("http://localhost:8002/image_encoder", data=image_bytes, headers=headers)
            
            text_features.append(np.asarray(json.loads(text_response.json())))
            image_features.append(np.asarray(json.loads(image_response.json())))

        text_features = np.concatenate(text_features)
        image_features = np.concatenate(image_features)

        return np.concatenate([text_features, image_features], axis=1)
