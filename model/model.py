import os
import json
import numpy as np
import cv2
import onnxruntime
import torch


threshold = 0.5
device = "cuda" if torch.cuda.is_available() else "cpu"


class Model():
    def __init__(self, model_file: str="classifier.onnx") -> None:
        self.onnx_session = onnxruntime.InferenceSession(model_file,
                                                         providers=['CUDAExecutionProvider',
                                                                    'CPUExecutionProvider'])


    def predict_on_imgs(self, image_names, text_responses, image_responses, batch_size=512):
        features = self.get_features(text_responses, image_responses).astype(np.float32)

        preds = []

        for i in range(0, len(features), batch_size):
            batch = features[i:min(i+batch_size, len(features))]
            preds.extend(self.onnx_session.run(None, {'input': batch})[0])

        preds = np.concatenate(preds).tolist()

        res = []
        for i, name in enumerate(image_names):
            res.append({"image": name, "score": preds[i], "target": preds[i] >= threshold})
        return res


    def get_features(self, text_responses, image_responses):
        """
        Get features from model

        Parameters
        ----------
        text_response
            response from text encoder
        image_response
            response from image encoder

        Returns
        -------
        np.ndarray
            2d array of concatenated images an text embeddings
        """

        text_features = np.concatenate([np.asarray(json.loads(text_response.json())) for text_response in text_responses])
        image_features = np.concatenate([np.asarray(json.loads(image_response.json())) for image_response in image_responses])

        return np.concatenate([text_features, image_features], axis=1)
