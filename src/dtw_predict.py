import numpy as np

from .dtw import dtw
from .silence import trim_silence


def predict(features: np.ndarray, templates: dict[str, np.ndarray]) -> tuple[str, dict[str, float]]:
    sequence = trim_silence(features)
    costs = {vowel: dtw(sequence, template)[0] for vowel, template in templates.items()}
    return min(costs, key=costs.get), costs
