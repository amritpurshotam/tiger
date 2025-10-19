from typing import NamedTuple

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn

from tiger.distributions.gumbel import gumbel_softmax_sample
from tiger.models.enums import QuantizeDistance, QuantizeGradientFlow
from tiger.models.kmeans import Kmeans
from tiger.models.loss import QuantizeLoss


class QuantizeOutput(NamedTuple):
    embeddings: Tensor
    ids: Tensor
    loss: Tensor


def efficient_rotation_trick_transform(u, q, e):
    """4.2 in https://arxiv.org/abs/2410.06424."""
    e = rearrange(e, "b d -> b 1 d")
    w = F.normalize(u + q, p=2, dim=1, eps=1e-6).detach()

    return (
        e
        - 2 * (e @ rearrange(w, "b d -> b d 1") @ rearrange(w, "b d -> b 1 d"))
        + 2 * (e @ rearrange(u, "b d -> b d 1").detach() @ rearrange(q, "b d -> b 1 d").detach())
    ).squeeze()


class Quantize(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_weight: float,
        gradient_flow_mode: QuantizeGradientFlow,
        distance_mode: QuantizeDistance,
    ):
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.gradient_flow_mode = gradient_flow_mode
        self.distance_mode = distance_mode

        self.quantize_loss = QuantizeLoss(commitment_weight)
        self.kmeans_initted = False
        self._init_weights()

    def forward(self, x: Tensor, temperature: float) -> QuantizeOutput:
        if not self.kmeans_initted:
            self._kmeans_init(x)

        codebook = self.embedding.weight

        if self.distance_mode == QuantizeDistance.EUCLIDEAN:
            dist = torch.cdist(x, codebook)
        elif self.distance_mode == QuantizeDistance.COSINE:
            dist = -(F.normalize(x, p=2, dim=1) @ F.normalize(codebook, p=2, dim=1).T)
        else:
            raise ValueError("Unsupported distance mode.")

        ids = torch.argmin(dist.detach(), dim=-1)

        if self.training:
            if self.gradient_flow_mode == QuantizeGradientFlow.GUMBEL_SOFTMAX:
                weights = gumbel_softmax_sample(-dist, temperature)
                emb = weights @ codebook
                emb_out = emb
            elif self.gradient_flow_mode == QuantizeGradientFlow.STE:
                emb = self.get_item_embeddings(ids)
                emb_out = x + (emb - x).detach()
            elif self.gradient_flow_mode == QuantizeGradientFlow.ROTATION_TRICK:
                emb = self.get_item_embeddings(ids)
                emb_out = efficient_rotation_trick_transform(
                    x / (x.norm(dim=-1, keepdim=True) + 1e-8),
                    emb / (emb.norm(dim=-1, keepdim=True) + 1e-8),
                    x,
                )
                emb_out = (
                    emb_out
                    * (
                        torch.norm(emb, dim=1, keepdim=True) / (torch.norm(x, dim=1, keepdim=True) + 1e-6)
                    ).detach()
                )
            else:
                raise ValueError("Unsupported Gradient Flow Mode.")

            loss = self.quantize_loss(x, emb)
        else:
            emb_out = self.get_item_embeddings(ids)
            loss = self.quantize_loss(x, emb_out)

        return QuantizeOutput(embeddings=emb_out, ids=ids, loss=loss)

    def get_item_embeddings(self, item_ids) -> Tensor:
        return self.embedding(item_ids)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight)

    def _kmeans_init(self, x):
        with torch.no_grad():
            k, _ = self.embedding.weight.shape
            kmeans_out = Kmeans(k).run(x)
            self.embedding.weight.data.copy_(kmeans_out.centroids)

        self.kmeans_initted = True
