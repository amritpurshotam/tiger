from torch import Tensor, nn

from tiger.models.normalize import L2NormalizationLayer


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], out_dim: int, normalize: bool = False):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = out_dim

        dims = [self.input_dim] + self.hidden_dims + [self.output_dim]

        self.mlp = nn.Sequential()
        for i, (in_d, out_d) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
            self.mlp.append(nn.Linear(in_d, out_d, bias=False))
            if i >= len(dims) - 2:
                continue

            self.mlp.append(nn.SiLU())

        self.mlp.append(L2NormalizationLayer() if normalize else nn.Identity())

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)
