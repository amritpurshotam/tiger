import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from sklearn.cluster import KMeans


class RQVAE(nn.Module):
    def __init__(self, input_dim=768, latent_dim=32, num_codebooks=3, codebook_size=256, beta=0.25):
        super(RQVAE, self).__init__()

        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.beta = beta

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
        )

        self.codebooks = nn.ModuleList(
            [nn.Embedding(codebook_size, latent_dim) for _ in range(num_codebooks)]
        )

    def quantize(self, level, residual):
        codebook = self.codebooks[level]
        distance = torch.cdist(residual, codebook.weight)
        semantic_id = torch.argmin(distance, dim=-1).squeeze()
        codeword_embedding = codebook(semantic_id)
        return codeword_embedding, semantic_id

    def initialize_codebooks(self, data_batch):
        # Perform K-means clustering on the first batch to initialize codebooks
        z = self.encoder(data_batch).detach().cpu().numpy()

        # Run K-means clustering for each codebook level to get centroids
        for codebook in self.codebooks:
            kmeans = KMeans(n_clusters=self.codebook_size, random_state=42)
            kmeans.fit(z)
            centroids = kmeans.cluster_centers_

            # Load centroids into codebook weights
            codebook.weight.data.copy_(torch.tensor(centroids, dtype=torch.float32))

            # Compute residual for the next level
            z = z - centroids[kmeans.predict(z)]

    def forward(self, x):
        codeword_embeddings = []
        semantic_ids = []
        residuals = []

        residual = self.encoder(x)

        for level in range(self.num_codebooks):
            residuals.append(residual)

            codeword_embedding, semantic_id = self.quantize(level, residual)
            residual -= codeword_embedding

            codeword_embeddings.append(codeword_embedding)
            semantic_ids.append(semantic_id)

        codeword_embeddings = torch.stack(codeword_embeddings, dim=-1)
        semantic_ids = torch.stack(semantic_ids, dim=-1)
        residuals = torch.stack(residuals, dim=-1)

        z_hat = codeword_embeddings.sum(axis=-1)
        x_hat = self.decoder(z_hat)

        recon_loss = F.mse_loss(x_hat, x, reduction="none").sum(axis=-1)  # calculate loss per sample in batch
        rqvae_loss = self.compute_rqvae_loss(codeword_embeddings, residuals)
        loss = (recon_loss + rqvae_loss).mean()  # take mean across batch

        return x_hat, loss

    def compute_rqvae_loss(self, codeword_embeddings, residuals):
        loss = 0
        loss += (  # sum across codebooks then latent_dim
            F.mse_loss(codeword_embeddings, residuals.detach(), reduction="none").sum(axis=[-1, -2])
            + self.beta
            * F.mse_loss(codeword_embeddings.detach(), residuals, reduction="none").sum(axis=[-1, -2])
        )
        return loss
