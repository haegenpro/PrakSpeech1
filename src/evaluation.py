import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import dtw_predict, gaussian
from .dataset_utils import VOWELS, load_dataset
from .dtw_templates import build_templates
from .gaussian import build_models

SYSTEMS = ("dtw", "gaussian")
SCENARIOS = ("closed_recording", "closed_speakers", "open_speakers")
SCENARIO_PARTITION = {"closed_recording": "template", "closed_speakers": "closed", "open_speakers": "open"}
SYSTEM_LABEL = {"dtw": "DTW Template", "gaussian": "Model Gaussian"}
SCENARIO_LABEL = {"closed_recording": "Closed Recording", "closed_speakers": "Closed Speakers", "open_speakers": "Open Speakers"}


def run_experiments(features_dir="features", progress=None) -> list[dict]:
    template_data = load_dataset("template", features_dir)
    dtw_models = build_templates(template_data)
    gauss_models = build_models(template_data)
    predictors = {"dtw": lambda f: dtw_predict.predict(f, dtw_models),
                  "gaussian": lambda f: gaussian.predict(f, gauss_models)}
    rows = []
    for scenario, partition in SCENARIO_PARTITION.items():
        for item in load_dataset(partition, features_dir):
            for system in SYSTEMS:
                pred, costs = predictors[system](item["features"])
                rows.append({"system": system, "scenario": scenario, "speaker": item["speaker"],
                             "true": item["vowel"], "pred": pred,
                             "correct": int(pred == item["vowel"]),
                             "file": Path(item["file_path"]).name,
                             **{f"cost_{v}": costs[v] for v in VOWELS}})
            if progress:
                progress(scenario, item["file_path"])
    return rows


def save_predictions(rows: list[dict], path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_predictions(path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["correct"] = int(r["correct"])
    return rows


def accuracy(rows: list[dict], **where) -> float:
    sel = [r for r in rows if all(r[k] == v for k, v in where.items())]
    if not sel:
        return float("nan")
    return 100.0 * sum(r["correct"] for r in sel) / len(sel)


def _mean(values):
    values = [v for v in values if not np.isnan(v)]
    return float(np.mean(values)) if values else float("nan")


def summary_table(rows):
    out = []
    for s in SYSTEMS:
        accs = [accuracy(rows, system=s, scenario=sc) for sc in SCENARIOS]
        out.append([SYSTEM_LABEL[s], *accs, _mean(accs)])
    return ["Sistem", *[SCENARIO_LABEL[sc] for sc in SCENARIOS], "Rata-rata"], out


def per_vowel_table(rows):
    cols = [(s, sc) for s in SYSTEMS for sc in SCENARIOS]
    header = ["Vowel", *[f"{'DTW' if s == 'dtw' else 'Gaussian'} {SCENARIO_LABEL[sc]}"
                         for s, sc in cols]]
    out = []
    for v in VOWELS:
        out.append([v, *[accuracy(rows, system=s, scenario=sc, true=v) for s, sc in cols]])
    out.append(["Average Accuracy", *[_mean([r[i + 1] for r in out]) for i in range(len(cols))]])
    return header, out


def per_speaker_table(rows, scenarios, label):
    cols = [(s, sc) for sc in scenarios for s in SYSTEMS]
    header = [label, *[f"{'DTW' if s == 'dtw' else 'Gaussian'} {SCENARIO_LABEL[sc]}"
                       for s, sc in cols], "Average"]
    speakers = sorted({r["speaker"] for r in rows if r["scenario"] in scenarios})
    out = []
    for sp in speakers:
        accs = [accuracy(rows, system=s, scenario=sc, speaker=sp) for s, sc in cols]
        out.append([sp, *accs, _mean(accs)])
    out.append(["Average", *[_mean([r[i + 1] for r in out]) for i in range(len(cols) + 1)]])
    return header, out


def confusion_matrix(rows, system, scenario):
    idx = {v: i for i, v in enumerate(VOWELS)}
    m = np.zeros((len(VOWELS), len(VOWELS)), dtype=int)
    for r in rows:
        if r["system"] == system and r["scenario"] == scenario:
            m[idx[r["true"]], idx[r["pred"]]] += 1
    return m


def error_list(rows):
    return [r for r in rows if not r["correct"]]


def top_confusions(rows, system, scenario=None):
    counts = defaultdict(int)
    for r in rows:
        if r["system"] == system and not r["correct"] and (scenario is None or r["scenario"] == scenario):
            counts[(r["true"], r["pred"])] += 1
    return sorted(counts.items(), key=lambda kv: -kv[1])


def fmt(x):
    return "-" if isinstance(x, float) and np.isnan(x) else (f"{x:.1f}%" if isinstance(x, float) else str(x))


def to_markdown(header, body) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(fmt(c) for c in row) + " |" for row in body]
    return "\n".join(lines)


def write_csv_table(header, body, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in body:
            w.writerow([f"{c:.2f}" if isinstance(c, float) else c for c in row])