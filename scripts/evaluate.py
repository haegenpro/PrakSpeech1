import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import evaluation as ev
from src.dataset_utils import VOWELS


def confusion_md(rows, system, scenario):
    m = ev.confusion_matrix(rows, system, scenario)
    head = "| benar \\ prediksi | " + " | ".join(VOWELS) + " |\n|" + "---|" * (len(VOWELS) + 1)
    body = "\n".join(f"| {v} | " + " | ".join(str(x) for x in m[i]) + " |"
                     for i, v in enumerate(VOWELS))
    return head + "\n" + body


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features-dir", default="features")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--reuse", action="store_true")
    a = p.parse_args()
    out = Path(a.out_dir)
    pred_path = out / "predictions.csv"
    if a.reuse:
        rows = ev.load_predictions(pred_path)
    else:
        rows = ev.run_experiments(a.features_dir, progress=lambda s, f: print(s, Path(f).name, flush=True))
        ev.save_predictions(rows, pred_path)

    tables = {
        "table0_summary": ev.summary_table(rows),
        "table1_per_vowel": ev.per_vowel_table(rows),
        "table2_closed_per_speaker": ev.per_speaker_table(rows, ("closed_recording", "closed_speakers"), "Speaker"),
        "table3_open_per_speaker": ev.per_speaker_table(rows, ("open_speakers",), "Speaker (open)"),
    }
    md = ["# Hasil Evaluasi (Person 4)\n"]
    titles = {"table0_summary": "Ringkasan akurasi (%)",
              "table1_per_vowel": "Tabel 1: Akurasi per vowel",
              "table2_closed_per_speaker": "Tabel 2: Closed experiments per speaker",
              "table3_open_per_speaker": "Tabel 3: Open experiments per speaker"}
    for name, (h, b) in tables.items():
        ev.write_csv_table(h, b, out / f"{name}.csv")
        md += [f"## {titles[name]}\n", ev.to_markdown(h, b), ""]
    md.append("## Confusion matrix\n")
    for s in ev.SYSTEMS:
        for sc in ev.SCENARIOS:
            md += [f"### {ev.SYSTEM_LABEL[s]} - {ev.SCENARIO_LABEL[sc]}\n", confusion_md(rows, s, sc), ""]
    md.append("## Daftar kesalahan\n")
    md.append("| Sistem | Skenario | Speaker | Benar | Prediksi | File |\n|---|---|---|---|---|---|")
    for r in ev.error_list(rows):
        md.append(f"| {ev.SYSTEM_LABEL[r['system']]} | {ev.SCENARIO_LABEL[r['scenario']]} | "
                  f"{r['speaker']} | {r['true']} | {r['pred']} | {r['file']} |")
    (out / "evaluation_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()