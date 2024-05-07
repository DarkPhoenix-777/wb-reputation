import os
from typing import Dict, List
import numpy as np
import torch
import easyocr


device = "cuda" if torch.cuda.is_available() else "cpu"

class OCR():
    """get image + text embeddings"""
    def __init__(self, reader = easyocr.Reader(lang_list=["en", "ru"])) -> None:

        self.reader = reader

        if device == "cpu":
            print("Warning: CUDA not detected by torch, using CPU")
    
    def read_text(self, images: List[np.ndarray], batch_size=64) -> List[str]:
        """
        Read text from image

        Parameters
        ----------
        images : List[np.ndarray]
            images in RGB
        batch_size : int
            batch size for EasyOCR 

        Returns
        -------
        List[str]
            texts from image
        """
        texts = []
        for image in images:
            texts.append(" ".join(self.reader.readtext(image,
                                                       rotation_info=[90, 270],
                                                       detail=0,
                                                       batch_size=batch_size)))
        return texts
