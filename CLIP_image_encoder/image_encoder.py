import os
from typing import Dict, List
import numpy as np
import torch
from transformers import CLIPImageProcessor, CLIPModel


device = "cuda" if torch.cuda.is_available() else "cpu"

class Image_encoder():
    """get image embeddings"""
    def __init__(self,
                 image_preprocessor=CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32"),
                 image_model=CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()) -> None:

        self.image_preprocessor = image_preprocessor
        self.image_model = image_model

        if device == "cpu":
            print("Warning: CUDA not detected by torch, using CPU")
    
    def get_img_embeddings(self, imgs_list = List[np.ndarray], batch_size=64):
        """
        Get image embeddings

        Parameters
        ----------
        imgs_list : List[np.ndarray]
            list of images in RGB
        batch_size : int
            batch size for image model

        Returns
        -------
        np.ndarray
            2d array of images embeddings
        """
        image_features = []
        with torch.no_grad():
            for i in range(0, len(imgs_list), batch_size):
                batch = imgs_list[i:min(i+batch_size, len(imgs_list))]
                inputs = self.image_preprocessor(images=batch, return_tensors="pt")
                image_features.append(self.image_model.get_image_features(**inputs.to(device)).cpu().numpy())

        return np.concatenate(image_features)
