import torch
import torch.nn as nn
from torch import Tensor


class NgramItemEmbedding(nn.Module):
    def __init__(
        self,
        codebook_size: int,
        n: int,
        num_embed: int,
        embed_dim: int,
    ):
        super(NgramItemEmbedding, self).__init__()

        self.codebook_size = codebook_size
        self.n = n

        self.powers = torch.zeros(self.n, self.n, dtype=torch.int32)
        for i in range(self.n):
            self.powers[i, : i + 1] = self.codebook_size ** torch.arange(
                start=i, end=-1, step=-1, dtype=torch.int32
            )

        self.embedding = nn.Embedding(
            num_embeddings=num_embed,
            embedding_dim=embed_dim,
        )

    def forward(self, x: Tensor):
        ngrams = self._calculate_ngrams(x)
        emb = self.embedding(ngrams).sum(dim=1)
        return emb

    def _calculate_ngrams(self, sem_id: Tensor) -> Tensor:
        sem_shifted = sem_id[:, : self.n] + 1
        ngrams = (sem_shifted[:, None, :] * self.powers[None, :, :]).sum(dim=-1) - 1
        return ngrams
