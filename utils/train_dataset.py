import numpy as np
import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, device: str) -> None:
        super().__init__()
        self.X = X
        self.y = y.reshape(-1, 1)
        self.device = device
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, ids):
        return torch.tensor(self.X[ids], dtype=torch.float32, device=self.device),\
            torch.tensor(self.y[ids], dtype=torch.float32, device=self.device)