import os
from typing import Dict, List
import numpy as np
import torch
from transformers import CLIPImageProcessor
import onnxruntime


class ImageEncoder():
    """get image embeddings"""
    def __init__(self,
                 image_preprocessor=CLIPImageProcessor\
                                    .from_pretrained("openai/clip-vit-base-patch32"),
                 image_model_file:str = "models/models_onnx/clip_image_encoder.onnx",
                 providers: List[str] = None,
                 batch_size: int = 64) -> None:

        self.batch_size = batch_size if batch_size > 0 else 64
        self.image_preprocessor = image_preprocessor
        self.onnx_session = onnxruntime.InferenceSession(image_model_file,
                                                         providers=providers)


    def get_img_embeddings(self, imgs_list = List[np.ndarray]):
        """
        Get image embeddings

        Parameters
        ----------
        imgs_list : List[np.ndarray]
            list of images in RGB

        Returns
        -------
        np.ndarray
            2d array of images embeddings
        """
        image_features = []

        for i in range(0, len(imgs_list), self.batch_size):
            batch = imgs_list[i:min(i+self.batch_size, len(imgs_list))]
            inputs = self.image_preprocessor(images=batch)["pixel_values"]
            image_features.append(self.onnx_session.run(None, {'input': inputs})[0])

        return np.concatenate(image_features)
