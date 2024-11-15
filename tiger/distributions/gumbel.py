from typing import Tuple

import numpy as np
import torch


def sample_gumbel(shape: Tuple, eps=1e-20) -> torch.Tensor:
    U = torch.rand(shape, device="cuda")  # noqa: N806
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    y = logits + sample_gumbel(logits.shape)
    return torch.nn.functional.softmax(y / temperature, dim=-1)


class TemperatureScheduler:
    def __init__(
        self,
        t0: float,
        min_temp: float,
        anneal_rate: float,
        step_size: int,
    ) -> None:
        self.t0 = t0
        self.min_temp = min_temp
        self.anneal_rate = anneal_rate
        self.step_size = step_size
        self.temp = t0

    def update_temp(self, epoch: int):
        if epoch % self.step_size == self.step_size - 1:
            self.temp = np.maximum(self.temp * np.exp(-self.anneal_rate * epoch), self.min_temp)

    def get_temp(self, epoch: int):
        self.update_temp(epoch)
        return self.temp
