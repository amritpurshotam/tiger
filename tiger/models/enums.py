from enum import Enum


class QuantizeGradientFlow(Enum):
    GUMBEL_SOFTMAX = 1
    STE = 2
    ROTATION_TRICK = 3


class QuantizeDistance(Enum):
    EUCLIDEAN = 1
    COSINE = 2
