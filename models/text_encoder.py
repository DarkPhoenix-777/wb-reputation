import os
from typing import Dict, List
import numpy as np
import torch
from transformers import DistilBertTokenizer
import onnxruntime


class TextEncoder():
    """get text embeddings"""
    def __init__(self,
                 text_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased"),
                 text_model_file: str = "models/models_onnx/distilbert-base-uncased.onnx",
                 providers: List[str] = None,
                 batch_size: int = 64) -> None:

        self.batch_size = batch_size if batch_size > 0 else 64
        self.tokenizer = text_tokenizer
        self.onnx_session = onnxruntime.InferenceSession(text_model_file,
                                                         providers=providers)


    def tokenize_text(self, batch: List[str]) -> List[np.ndarray]:
        """
        Tokenize text from list of str

        Parameters
        ----------
        batch : List[str]
            list of texts

        Returns
        -------
        List[np.ndarray]
            list of tokenized texts
        """
        max_length = 256
        tokenized_texts = []

        for text in batch:
            tokenized_text = self.tokenizer.encode(text, add_special_tokens=True, 
                                                   max_length=max_length, truncation=True)
            tokenized_text += [0 for _ in range(max_length - len(tokenized_text))]
            tokenized_texts.append(tokenized_text)

        return np.array(tokenized_texts)


    def get_embedding(self, tokens: np.ndarray) -> np.ndarray:
        """
        Get text embeddings for tokens

        Parameters
        ----------
        tokens : np.ndarray
            2d array of tokens

        Returns
        -------
        np.ndarray
            2d array of text embeddings
        """
        mask = (tokens > 0).astype(np.int64)

        last_hidden_states = self.onnx_session.run(["last_hidden_state"],
                                                   {"input_ids": tokens.astype(np.int64),
                                                     "attention_mask": mask})

        features = last_hidden_states[0][:, 0, :]

        return features


    def get_text_embeddings(self, texts_list: List[str]) -> np.ndarray:
        """
        Get text embeddings for texts from images

        Parameters
        ----------
        texts_list : List[str]
            list of texts

        Returns
        -------
        np.ndarray
            2d array of text embeddings
        """
        text_features = []

        for i in range(0, len(texts_list), self.batch_size):
            batch = texts_list[i:min(i+self.batch_size, len(texts_list))]
            batch = self.tokenize_text(batch)
            text_features.append(self.get_embedding(np.array(batch)))

        return np.concatenate(text_features)
