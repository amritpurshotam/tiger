import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from tiger import config


class T5Recommender(nn.Module):
    def __init__(
        self,
        num_encoders: int,
        num_decoders: int,
        num_heads: int,
        head_dim: int,
        ff_dim: int,
        dropout: float,
        num_user_bins: int,
        codebook_size: int,
        num_codebooks: int,
        token_dim: int,
    ):
        super(T5Recommender, self).__init__()

        self.transformer = nn.Transformer(
            d_model=num_heads * head_dim,
            nhead=num_heads,
            num_encoder_layers=num_encoders,
            num_decoder_layers=num_decoders,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation=F.relu,
        )

        vocab_size = num_user_bins + (codebook_size * (num_codebooks + 1))
        self.emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=token_dim)

    def forward(self, x: Tensor):
        return self.transformer(x)


if __name__ == "__main__":
    model = T5Recommender(
        num_encoders=config.NUM_ENCODERS,
        num_decoders=config.NUM_DECODERS,
        num_heads=config.NUM_HEADS,
        head_dim=config.HEAD_DIM,
        ff_dim=config.FF_DIM,
        dropout=config.DROPOUT,
        num_user_bins=config.USER_NUM_BINS,
        codebook_size=config.CODEBOOK_SIZE,
        num_codebooks=config.NUM_CODEBOOKS,
        token_dim=config.TOKEN_DIM,
    )

    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
