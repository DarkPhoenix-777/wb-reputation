import os
from typing import Dict, List
import numpy as np
import torch
import easyocr


class OCR():
    """read texts from image"""
    def __init__(self, reader = easyocr.Reader(lang_list=["en", "ru"]),
                 batch_size: int = 64) -> None:

        self.batch_size = batch_size #if batch_size > 0 else 64
        self.reader = reader

    def read_text(self, images: List[np.ndarray]) -> List[str]:
        """
        Read text from image

        Parameters
        ----------
        images : List[np.ndarray]
            images in RGB

        Returns
        -------
        List[str]
            texts from images
        """
        texts = []
        for image in images:
            texts.append(" ".join(self.reader.readtext(image,
                                                       rotation_info=[90, 270],
                                                       detail=0,
                                                       batch_size=self.batch_size)))
        return texts
