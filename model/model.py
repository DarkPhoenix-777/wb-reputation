import os
import json
import numpy as np
import cv2
import torch
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl

threshold = 0.5
device = "cuda" if torch.cuda.is_available() else "cpu"

class Classifier(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1280, 512),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=0.5),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        self.layers.eval()
        x = torch.tensor(x).to(device)
        with torch.no_grad():
            preds = self.layers(x)
        return np.concatenate(preds.cpu().numpy())


class Model():
    def __init__(self) -> None:
        self.model = Classifier.load_from_checkpoint(checkpoint_path="nn_model.ckpt")
        self.model.freeze()


    def predict_on_imgs(self, image_names, text_response, image_response):
        features = self.get_features(text_response, image_response).astype(np.float32)
        preds = self.model.forward(features).tolist()
        res = []
        for i, name in enumerate(image_names):
            res.append({"image": name, "score": preds[i], "target": preds[i] >= threshold})
        return res

    def get_features(self, text_response, image_response):
        """
        Get features from model

        Parameters
        ----------

        Returns
        -------
        np.ndarray
            2d array of concatenated images an text embeddings
        """

        text_features = np.asarray(json.loads(text_response.json()))
        image_features = np.asarray(json.loads(image_response.json()))

        return np.concatenate([text_features, image_features], axis=1)
