import torch
import torch.nn as nn

class BinaryClassifier(nn.Module):
    def __init__(self, base_model, output_dim=1, embedding_dim=384):
        super().__init__()
        self.base_model = base_model
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, 256),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            torch.nn.Linear(64, output_dim)
        ) # Classification head

    def forward(self, x):
        x = self.base_model(x).last_hidden_state[:, 0, :]
        x = self.fc(x)
        return x