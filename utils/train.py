import os
import csv
from typing import Optional
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
import onnx
from onnx2torch import convert

from models.pipeline import Pipeline
from utils.train_dataset import Dataset

device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = int(os.environ.get("CLASSIFIER_TRAIN_BATCH_SIZE", 512))

class Classifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        onnx_model = onnx.load("models/models_onnx/classifier.onnx")
        torch_model = convert(onnx_model)

        # берём слои с весами из обученной модели
        self.layers = nn.Sequential(
            getattr(torch_model, "layers/layers/0/Gemm"),
            getattr(torch_model, "layers/layers/1/LeakyRelu"),
            nn.Dropout(p=0.5),
            getattr(torch_model, "layers/layers/3/Gemm"),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


class Trainer():
    """
    Дообучение классификатора

    Parameters
    ----------
    pipeline: Pipeline
        pipeline для извлечения признаков
    n_epoch: int
        Максимальное количество эпох
    early_stopping_rounds: Optional[int]
        Если None, то будет обучаться заданное количество эпох
        Если int, то остановит обучение, если лосс на валидации не уменьшается заданное количество эпох
    learning_rate: float
        Скорость обучения
    log_batch: bool
        Если True - выводит лосс на обучении по батчу
    
    """
    def __init__(self,
                 pipeline: Pipeline,
                 n_epoch: int = 10,
                 early_stopping_rounds: Optional[int] = None,
                 learning_rate: float = 1e-3,
                 log_batch: bool = False) -> None:
        
        self.pipeline = pipeline
        self.n_epoch = n_epoch
        self.early_stopping_rounds = early_stopping_rounds
        self.log_batch = log_batch

        self.model = Classifier().to(device)
        self.loss_func = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        images = []
        labels = []
        subset = []

        with open("data/images_data.csv", "r", newline="", encoding="utf-8") as file:
            reader = csv.reader(file)

            for i, row in enumerate(reader):
                if i == 0:
                    continue
                image_name = row[0]
                if os.path.isfile("data/images/" + image_name):
                    images.append(Image.open("data/images/" + image_name))
                    labels.append(int(row[1]))
                    subset.append(row[2])
                else:
                    print(f"Image {image_name} not found. Passing")

        labels = np.array(labels)
        subset = np.array(subset)

        features = self.pipeline.get_features_from_images(images)

        X_train = features[subset == "t"]
        y_train = labels[subset == "t"]

        X_val = features[subset == "v"]
        y_val = labels[subset == "v"]

        self.train_loader = DataLoader(Dataset(X_train, y_train, device), batch_size=BATCH_SIZE, shuffle=True)
        self.val_loader = DataLoader(Dataset(X_val, y_val, device), batch_size=BATCH_SIZE)

    def train_one_epoch(self):
        running_loss = 0.

        for i, data in enumerate(self.train_loader):
            inputs, labels = data

            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            loss = self.loss_func(outputs, labels)
            loss.backward()

            self.optimizer.step()

            running_loss += loss.item()
            if self.log_batch:
                print(f'batch {i + 1} loss: {loss.item()}')

        return running_loss / len(self.train_loader)


    def train(self):
        patience = 0
        best_vloss = float('inf')

        for epoch in range(self.n_epoch):
            print(f'EPOCH {epoch + 1}:')

            self.model.train(True)
            avg_loss = self.train_one_epoch()

            running_vloss = 0.0
            self.model.eval()

            with torch.no_grad():
                for i, vdata in enumerate(self.val_loader):
                    vinputs, vlabels = vdata
                    voutputs = self.model(vinputs)
                    vloss = self.loss_func(voutputs, vlabels)
                    running_vloss += vloss

            avg_vloss = running_vloss / len(self.val_loader)
            print(f"LOSS train {avg_loss} valid {avg_vloss}")

            # проверка на увеличение лосса
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                patience = 0
            else:
                patience += 1

            if self.early_stopping_rounds is not None \
                and patience >= self.early_stopping_rounds:
                print(f"Early stopping triggered on epoch {epoch + 1}")
                break

    def save_classifier(self, save_path: str):
        """сохранение классификатора в формате onnx"""
        model = self.model.to("cpu")
        model.eval()
        with torch.no_grad():
            torch.onnx.export(model, torch.randn(1, 1280), save_path + "classifier.onnx",
                                input_names = ['input'],
                                verbose=False, export_params=True,
                                dynamic_axes={'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}})

