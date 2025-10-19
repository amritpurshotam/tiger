from torch import Tensor
from torch.utils.data import Dataset


class SentenceEmbeddingsDataset(Dataset):
    def __init__(self, data: Tensor):
        self.data = data

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx) -> Tensor:
        return self.data[idx]
