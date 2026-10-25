from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular

from .silence import trim_silence


def fit_gaussian(frames: np.ndarray, eps: float = 1e-6) -> dict[str, np.ndarray]:
    mean = frames.mean(axis=0)
    centered = frames - mean
    cov = centered.T @ centered / max(len(frames) - 1, 1)
    cov = cov + eps * np.trace(cov) / len(cov) * np.eye(len(cov))
    return {"mean": mean, "cov": cov}


def nll(frames: np.ndarray, model: dict[str, np.ndarray]) -> float:
    chol = np.linalg.cholesky(model["cov"])
    whitened = solve_triangular(chol, (frames - model["mean"]).T, lower=True)
    logdet = 2 * np.log(np.diag(chol)).sum()
    per_frame = 0.5 * (len(chol) * np.log(2 * np.pi) + logdet + (whitened ** 2).sum(axis=0))
    return float(per_frame.mean())


def build_models(records: list[dict], eps: float = 1e-6,
                 trim: bool = True) -> dict[str, dict[str, np.ndarray]]:
    groups = {}
    for record in records:
        features = trim_silence(record["features"]) if trim else record["features"]
        groups.setdefault(record["vowel"], []).append(features)
    return {vowel: fit_gaussian(np.concatenate(sequences), eps)
            for vowel, sequences in sorted(groups.items())}


def predict(features: np.ndarray, models: dict[str, dict[str, np.ndarray]],
            trim: bool = True) -> tuple[str, dict[str, float]]:
    sequence = trim_silence(features) if trim else features
    costs = {vowel: nll(sequence, model) for vowel, model in models.items()}
    return min(costs, key=costs.get), costs


def save_models(models: dict[str, dict[str, np.ndarray]], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{f"{vowel}_{name}": value
                      for vowel, model in models.items() for name, value in model.items()})


def load_models(path: str | Path) -> dict[str, dict[str, np.ndarray]]:
    with np.load(path) as data:
        vowels = sorted({name.split("_")[0] for name in data.files})
        return {vowel: {"mean": data[f"{vowel}_mean"], "cov": data[f"{vowel}_cov"]}
                for vowel in vowels}
