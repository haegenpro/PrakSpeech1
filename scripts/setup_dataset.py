"""Consolidate recordings, verify hashes, and remove duplicate legacy originals."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset_utils import setup_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="features")
    args = parser.parse_args()
    print(json.dumps(setup_dataset(args.data_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
