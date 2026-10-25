"""Extract indexed 39-dimensional frame sequences for all available WAVs."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset_utils import extract_dataset_features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="features")
    args = parser.parse_args()
    summary = extract_dataset_features(args.data_dir, args.output_dir)
    print(json.dumps(summary, indent=2))
    sys.exit(1 if summary["failed"] else 0)


if __name__ == "__main__":
    main()
