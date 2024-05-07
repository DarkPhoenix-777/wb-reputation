import os
from typing import Dict, List
import numpy as np
import torch
from transformers import DistilBertModel, DistilBertTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

class Text_encoder():
    """get text embeddings"""
    def __init__(self,
                 text_model_name: str = "distilbert-base-uncased") -> None:

        self.tokenizer = DistilBertTokenizer.from_pretrained(text_model_name)
        self.bert = DistilBertModel.from_pretrained(text_model_name).to(device)

        if device == "cpu":
            print("Warning: CUDA not detected by torch, using CPU")


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
            tokenized_text = self.tokenizer.encode(text, add_special_tokens=True, max_length=max_length, truncation=True)
            tokenized_text += [0 for _ in range(max_length - len(tokenized_text))]
            tokenized_texts.append(tokenized_text)

        return tokenized_texts


    def get_embedding(self, tokens: np.ndarray):
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
        mask = tokens > 0

        tokens = torch.tensor(tokens).to(device)
        mask = torch.tensor(mask).to(device)

        with torch.no_grad():
            last_hidden_states = self.bert(tokens, attention_mask=mask)

        features = last_hidden_states[0][:, 0, :].cpu().numpy()

        return features


    def get_text_embeddings(self, texts_list: List[str], batch_size=64) -> np.ndarray:
        """
        Get text embeddings for texts from images

        Parameters
        ----------
        texts_list : List[str]
            list of texts
        batch_size : int
            batch size for text model

        Returns
        -------
        np.ndarray
            2d array of text embeddings
        """
        text_features = []
        for i in range(0, len(texts_list), batch_size):
            batch = texts_list[i:min(i+batch_size, len(texts_list))]
            batch = self.tokenize_text(batch)
            text_features.append(self.get_embedding(np.array(batch)))
        return np.concatenate(text_features)
