from pathlib import Path

import numpy as np

from .dtw import dtw
from .silence import trim_silence


def average_template(sequences: list[np.ndarray]) -> np.ndarray:
    totals = [sum(dtw(a, b)[0] for b in sequences) for a in sequences]
    reference = sequences[int(np.argmin(totals))]
    summed = np.zeros(reference.shape)
    counts = np.zeros(len(reference))
    for sequence in sequences:
        for i, j in dtw(reference, sequence)[1]:
            summed[i] += sequence[j]
            counts[i] += 1
    return summed / counts[:, None]


def build_templates(records: list[dict]) -> dict[str, np.ndarray]:
    groups = {}
    for record in records:
        groups.setdefault(record["vowel"], []).append(trim_silence(record["features"]))
    return {vowel: average_template(sequences) for vowel, sequences in sorted(groups.items())}


def save_templates(templates: dict[str, np.ndarray], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **templates)


def load_templates(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {vowel: data[vowel] for vowel in data.files}
