import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

class EmotionClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super(EmotionClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2)
        )

    def forward(self, x):
        return self.model(x)

def createmodel(X, path):
    X = np.array(X, dtype = np.float32)
    model = EmotionClassifier(input_size=X.shape[-1])
    dict = torch.load(path)
    model.load_state_dict(dict)

    Xtest = torch.tensor(X, dtype = torch.float32)
    with torch.no_grad():
        predicted = model(Xtest).detach().numpy()
        val_pred = predicted[0]
        ars_pred = predicted[1]
    return val_pred, ars_pred
