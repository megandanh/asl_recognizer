import torch
import torch.nn as nn
import torch.nn.functional as F
 
class AutoEncoderClassifier(nn.Module):
    # This class will serve as the simplified CNN which
    # takes the Autoencoder's latent feature space as input.
    def __init__(self, latent_dim=256, num_classes=24):
        super(AutoEncoderClassifier, self).__init__()
        # This model uses 3 fully connected layers
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, X):
        # The data is passed simply through the 3 layers
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)  # Output layer (class scores)
        return X
