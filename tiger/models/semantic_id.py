import torch.nn as nn


class RQVAE(nn.Module):
    def __init__(self, input_dim=768, latent_dim=32):
        super(RQVAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return z
