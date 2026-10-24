import numpy as np
from scipy.spatial.distance import cdist


def dtw(a: np.ndarray, b: np.ndarray) -> tuple[float, list[tuple[int, int]]]:
    cost = cdist(a, b)
    n, m = cost.shape
    total = np.full((n + 1, m + 1), np.inf)
    total[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            total[i, j] = cost[i - 1, j - 1] + min(total[i - 1, j - 1], total[i - 1, j], total[i, j - 1])
    path = backtrack(total)
    return float(total[n, m]) / len(path), path


def backtrack(total: np.ndarray) -> list[tuple[int, int]]:
    i, j = total.shape[0] - 1, total.shape[1] - 1
    path = [(i - 1, j - 1)]
    while i > 1 or j > 1:
        i, j = min([(i - 1, j - 1), (i - 1, j), (i, j - 1)], key=lambda step: total[step])
        path.append((i - 1, j - 1))
    return path[::-1]
