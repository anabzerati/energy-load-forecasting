from torch.utils.data import Dataset
import torch

class TimeSeriesDataset(Dataset):
    """
    Custom PyTorch Dataset for time series forecasting.

    Args:
        X (np.ndarray): Input features of shape (samples, timesteps, features)
        y (np.ndarray): Target values of shape (samples, 1) or (samples,)

    """
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
