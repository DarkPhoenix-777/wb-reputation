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
    
    def read_text(self, img: np.ndarray, batch_size=64) -> str:
        """
        Read text from image

        Parameters
        ----------
        img : np.ndarray
            image in RGB
        batch_size : int
            batch size for EasyOCR 

        Returns
        -------
        str
            text from image
        """
        text = self.reader.readtext(img, rotation_info=[90, 270], detail=0, batch_size=batch_size)
        return " ".join(text)
