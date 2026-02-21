from enum import StrEnum


class QuantizeGradientFlow(StrEnum):
    GUMBEL_SOFTMAX = "gumbel"
    STE = "ste"
    ROTATION_TRICK = "rotation_trick"


class QuantizeDistance(StrEnum):
    EUCLIDEAN = "euclidean"
    COSINE = "cosine"
