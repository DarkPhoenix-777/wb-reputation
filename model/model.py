import os
import numpy as np
import cv2
import torch
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from features_extractor import FeatureExtractor

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
        self.feature_extractor = FeatureExtractor()
        self.model = Classifier.load_from_checkpoint(checkpoint_path="nn_model.ckpt")
        self.model.freeze()


    def predict_on_imgs(self, images, image_names, content_types):
        features = self.feature_extractor.get_features(images, image_names, content_types).astype(np.float32)
        preds = self.model.forward(features).tolist()
        res = []
        for i in range(len(images)):
            res.append({"image": image_names[i], "score": preds[i], "target": preds[i] >= threshold})
        return res
