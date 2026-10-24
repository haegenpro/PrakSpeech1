import numpy as np


def trim_silence(features: np.ndarray, threshold_db: float = 20) -> np.ndarray:
    energy = features[:, 0]
    limit = energy.max() - threshold_db * np.log(10) / 10
    active = np.flatnonzero(energy >= limit)
    return features[active[0]:active[-1] + 1]
