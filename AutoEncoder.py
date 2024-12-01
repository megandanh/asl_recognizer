import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoEncoder(nn.Module):
    def __init__(self, input_shape=(3, 28, 28), num_classes=24):
        super(AutoEncoder, self).__init__()

        # Encoder
        self.encoder_conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.encoder_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.encoder_conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Compute flatten_dim
        self.flatten_dim = 128 * 7 * 7  # Match encoder output size

        # Latent representation
        self.latent_dim = 256
        self.fc1 = nn.Linear(self.flatten_dim, self.latent_dim)

        # Decoder: Fully connected layers followed by transpose convolutions
        self.fc2 = nn.Linear(self.latent_dim, 128 * 7 * 7)  # Adjusted to match (128, 7, 7)
        self.decoder_conv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)  # Upsample to (64, 14, 14)
        self.decoder_conv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)  # Upsample to (32, 28, 28)
        self.decoder_conv3 = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1)  # Keep (3, 28, 28)

    def forward(self, X):
        # Encoder
        X = F.relu(self.encoder_conv1(X))
        X = self.pool(F.relu(self.encoder_conv2(X)))
        X = self.pool(F.relu(self.encoder_conv3(X)))

        # Flatten
        X = X.view(X.size(0), -1)  # Flatten for the bottleneck
        encoded = F.relu(self.fc1(X))

        # Decoder
        X = F.relu(self.fc2(encoded))
        X = X.view(X.size(0), 128, 7, 7)  # Reshape for transpose convolutions
        X = F.relu(self.decoder_conv1(X))  # Upsample to (64, 14, 14)
        X = F.relu(self.decoder_conv2(X))  # Upsample to (32, 28, 28)
        X = self.decoder_conv3(X)  # Final output (3, 28, 28)

        return X, encoded  # Return both the reconstructed image and the latent vector
