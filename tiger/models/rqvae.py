from typing import NamedTuple

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from tiger.models.encoder import MLP
from tiger.models.enums import QuantizeDistance, QuantizeGradientFlow
from tiger.models.loss import ReconstructionLoss
from tiger.models.quantize import Quantize, QuantizeOutput


class RqVaeOutput(NamedTuple):
    embeddings: Tensor
    residuals: Tensor
    sem_ids: Tensor
    quantize_loss: Tensor


class RqVaeComputedLosses(NamedTuple):
    loss: Tensor
    reconstruction_loss: Tensor
    rqvae_loss: Tensor
    embs_norm: Tensor
    p_unique_ids: Tensor


class RQVAE(nn.Module):
    def __init__(
        self,
        num_codebooks: int,
        codebook_size: int,
        latent_dim: int,
        beta: float,
        distance_mode: QuantizeDistance,
        gradient_flow_mode: QuantizeGradientFlow,
        input_dim: int,
        hidden_dims: list[int],
        normalize: bool,
    ):
        super(RQVAE, self).__init__()

        self.encoder = MLP(input_dim, hidden_dims, latent_dim, normalize)
        self.decoder = MLP(latent_dim, hidden_dims[::-1], input_dim, normalize)

        self.codebooks = nn.ModuleList(
            [
                Quantize(
                    num_embeddings=codebook_size,
                    embedding_dim=latent_dim,
                    commitment_weight=beta,
                    gradient_flow_mode=gradient_flow_mode,
                    distance_mode=distance_mode,
                )
                for i in range(num_codebooks)
            ]
        )

        self.recon_loss = ReconstructionLoss()

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def decode(self, x: Tensor) -> Tensor:
        return self.decoder(x)

    def get_semantic_ids(self, x: Tensor, temperature: float = 1e-3) -> RqVaeOutput:
        res = self.encode(x)

        quantize_loss = torch.tensor(0)
        embs, residuals, sem_ids = [], [], []

        for codebook in self.codebooks:
            residuals.append(res)
            quantized: QuantizeOutput = codebook(res, temperature)
            quantize_loss += quantized.loss
            emb, sem_id = quantized.embeddings, quantized.ids
            res = res - emb
            embs.append(emb)
            sem_ids.append(sem_id)

        return RqVaeOutput(
            embeddings=rearrange(embs, "c b d -> b d c"),  # type: ignore[arg-type]
            residuals=rearrange(residuals, "c b d -> b d c"),  # type: ignore[arg-type]
            sem_ids=rearrange(sem_ids, "c b -> b c"),  # type: ignore[arg-type]
            quantize_loss=quantize_loss,
        )

    def forward(self, x: Tensor, temperature: float):
        quantized = self.get_semantic_ids(x, temperature)
        embs = quantized.embeddings
        z_hat = embs.sum(dim=-1)
        x_hat = self.decode(z_hat)

        recon_loss = self.recon_loss(x_hat, x)
        codebook_loss = quantized.quantize_loss
        loss = (recon_loss + codebook_loss).mean()

        with torch.no_grad():
            # Compute debug ID statistics
            embs_norm = embs.norm(dim=1)
            p_unique_ids = (
                ~torch.triu(
                    (
                        rearrange(quantized.sem_ids, "b c -> b 1 c")
                        == rearrange(quantized.sem_ids, "b c -> 1 b c")
                    ).all(dim=-1),
                    diagonal=1,
                )
            ).all(dim=1).sum() / quantized.sem_ids.shape[0]

        return RqVaeComputedLosses(
            loss=loss,
            reconstruction_loss=recon_loss.mean(),
            rqvae_loss=codebook_loss.mean(),
            embs_norm=embs_norm,
            p_unique_ids=p_unique_ids,
        )
