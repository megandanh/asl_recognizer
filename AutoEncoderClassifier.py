import torch
import torch.nn as nn
import torch.nn.functional as F
class AutoEncoderClassifier(nn.Module):
    def __init__(self, latent_dim=256, num_classes=24):
        super(AutoEncoderClassifier, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)  # Output layer (class scores)
        return X