"""Leave-one-speaker-out grid on the template partition only."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset_utils import SPEAKERS, load_dataset
from src.gaussian import build_models, predict


def loso_correct(records: list[dict], trim: bool) -> int:
    correct = 0
    for held_out in SPEAKERS:
        train = [r for r in records if r["speaker"] != held_out]
        models = build_models(train, trim=trim)
        correct += sum(predict(r["features"], models, trim=trim)[0] == r["vowel"]
                       for r in records if r["speaker"] == held_out)
    return correct


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", default="features")
    args = parser.parse_args()
    records = load_dataset("template", args.features_dir)
    print("| Variasi | Benar (dari {}) |\n|---|---|".format(len(records)))
    for trim in (False, True):
        label = "Potong 20 dB" if trim else "Tanpa potong"
        print(f"| {label} | {loso_correct(records, trim)} |")


if __name__ == "__main__":
    main()
