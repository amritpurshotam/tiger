import torch.nn.functional as F
from torch import Tensor, nn


class ReconstructionLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x_hat: Tensor, x: Tensor) -> Tensor:
        loss = F.mse_loss(x_hat, x, reduction="none").sum(dim=-1)
        return loss


class QuantizeLoss(nn.Module):
    def __init__(self, commitment_weight: float = 0.25) -> None:
        super().__init__()
        self.commitment_weight = commitment_weight

    def forward(self, residual: Tensor, emb: Tensor) -> Tensor:
        codebook_loss = F.mse_loss(residual.detach(), emb, reduction="none").sum(dim=[-1])
        commitment_loss = F.mse_loss(residual, emb.detach(), reduction="none").sum(dim=[-1])

        return codebook_loss + self.commitment_weight * commitment_loss
