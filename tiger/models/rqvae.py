import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from sklearn.cluster import KMeans

from tiger.distributions.gumbel import gumbel_softmax_sample
from tiger.models.encoder import MLP


class RQVAE(nn.Module):
    def __init__(
        self,
        codebook_sizes: list[int],
        input_dim: int,
        hidden_dims: list[int],
        latent_dim: int,
        beta: float,
        normalize: bool,
        use_gumbel=False,
    ):
        super(RQVAE, self).__init__()

        self.num_codebooks = len(codebook_sizes)
        self.codebook_sizes = codebook_sizes
        self.beta = beta
        self.use_gumbel = use_gumbel

        self.encoder = MLP(input_dim, hidden_dims, latent_dim, normalize)
        self.decoder = MLP(latent_dim, hidden_dims[::-1], input_dim, normalize)

        self.codebooks = nn.ModuleList(
            [nn.Embedding(codebook_size, latent_dim) for codebook_size in self.codebook_sizes]
        )

    def quantize(self, level, residual, temperature):
        codebook = self.codebooks[level]
        distance = torch.cdist(residual, codebook.weight)
        semantic_id = torch.argmin(distance, dim=-1).squeeze()
        if self.train and self.use_gumbel:
            distance = gumbel_softmax_sample(-distance, temperature=temperature)
            codeword_embedding = distance @ codebook.weight
        else:
            codeword_embedding = codebook(semantic_id)
        return codeword_embedding, semantic_id

    def initialize_codebooks(self, data_batch):
        with torch.no_grad():
            z = self.encoder(data_batch).detach().cpu().numpy()

            for codebook in self.codebooks:
                kmeans = KMeans(n_clusters=codebook.weight.shape[0], random_state=42)
                kmeans.fit(z)
                centroids = kmeans.cluster_centers_

                codebook.weight.data.copy_(torch.tensor(centroids, dtype=torch.float32).to("cuda"))

                z = z - centroids[kmeans.predict(z)]

    def get_semantic_ids(self, x: torch.Tensor, temperature: float):
        codeword_embeddings_list = []
        semantic_ids_list = []
        residuals_list = []

        residual = self.encoder(x)

        for level in range(self.num_codebooks):
            residuals_list.append(residual)

            codeword_embedding, semantic_id = self.quantize(level, residual, temperature)
            residual = residual - codeword_embedding

            codeword_embeddings_list.append(codeword_embedding)
            semantic_ids_list.append(semantic_id)

        codeword_embeddings = torch.stack(codeword_embeddings_list, dim=-1)
        semantic_ids = torch.stack(semantic_ids_list, dim=-1)
        residuals = torch.stack(residuals_list, dim=-1)

        return codeword_embeddings, semantic_ids, residuals

    def forward(self, x, temperature):
        codeword_embeddings, _, residuals = self.get_semantic_ids(x, temperature)
        z_hat = codeword_embeddings.sum(axis=-1)
        x_hat = self.decoder(z_hat)

        recon_loss = F.mse_loss(x_hat, x, reduction="none").sum(axis=-1)  # calculate loss per sample in batch
        rqvae_loss = self.compute_rqvae_loss(codeword_embeddings, residuals)
        loss = (recon_loss + rqvae_loss).mean()  # take mean across batch

        return loss

    def compute_rqvae_loss(self, codeword_embeddings, residuals):
        loss = 0
        loss += (  # sum across codebooks then latent_dim
            F.mse_loss(codeword_embeddings, residuals.detach(), reduction="none").sum(axis=[-1, -2])
            + self.beta
            * F.mse_loss(codeword_embeddings.detach(), residuals, reduction="none").sum(axis=[-1, -2])
        )
        return loss
