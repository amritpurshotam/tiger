from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from tiger.models.enums import QuantizeDistance, QuantizeGradientFlow


@dataclass
class DatasetConfig:
    min_reviews_per_user: int
    sentence_model: str
    sentence_model_dim: int
    sequence_length: int
    amazon_year: int


@dataclass
class RqvaeConfig:
    num_codebooks: int
    codebook_size: int
    latent_dim: int
    beta: float
    hidden_dims: list[int]
    distance_mode: QuantizeDistance
    gradient_flow_mode: QuantizeGradientFlow
    normalize: bool

    def __post_init__(self):
        self.distance_mode = QuantizeDistance(self.distance_mode)
        self.gradient_flow_mode = QuantizeGradientFlow(self.gradient_flow_mode)


@dataclass
class RqvaeTrainConfig:
    batch_size: int
    lr: float
    max_lr: float
    num_epochs: int
    temp: float
    min_temp: float
    anneal_rate: float
    step_size: int
    val_split: int


@dataclass
class T5Config:
    num_encoders: int
    num_decoders: int
    num_heads: int
    head_dim: int
    ff_dim: int
    dropout: float
    token_dim: int
    user_num_bins: int
    padding_value: int


@dataclass
class Config:
    dataset: DatasetConfig
    rqvae: RqvaeConfig
    rqvae_train: RqvaeTrainConfig
    t5: T5Config

    @classmethod
    def from_yaml(cls, path: str | Path | None = None) -> "Config":
        with open("configs/base.yaml") as f:
            data = yaml.safe_load(f)
        if path is not None:
            with open(path) as f:
                overrides = yaml.safe_load(f) or {}
            for section, values in overrides.items():
                data[section].update(values)
        return cls(
            dataset=DatasetConfig(**data.get("dataset", {})),
            rqvae=RqvaeConfig(**data.get("rqvae", {})),
            rqvae_train=RqvaeTrainConfig(**data.get("rqvae_train", {})),
            t5=T5Config(**data.get("t5", {})),
        )

    def to_dict(self) -> dict:
        return asdict(self)
