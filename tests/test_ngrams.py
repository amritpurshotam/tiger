import torch
from torch.testing import assert_close

from tiger.models.ngram import NgramItemEmbedding


def test_ngram_calculation():
    ranker = NgramItemEmbedding(3, 256, 3)
    sem_id = [128, 190, 5]

    n_gram = ranker._calculate_ngram_iterative(sem_id)

    assert n_gram == [128, 33214, 8_503_045]


def test_ngram_fast_calculation():
    ranker = NgramItemEmbedding(3, 256, 3)
    sem_id = torch.tensor([[128, 190, 5], [0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0]])

    n_gram = ranker._calculate_ngrams(sem_id)

    expected_ngram = torch.tensor([[128, 33214, 8_503_045]])

    assert_close(n_gram, expected_ngram, atol=0, rtol=0)
