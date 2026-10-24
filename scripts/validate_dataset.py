"""Validate completeness, naming, audio decoding and partition separation."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset_utils import validate_dataset, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--report", type=Path, help="Also save the full JSON report")
    args = parser.parse_args()
    report = validate_dataset(args.data_dir)
    if args.report:
        write_json(args.report, report)
    print(json.dumps(report, indent=2))
    sys.exit(0 if report["valid"] else 1)


if __name__ == "__main__":
    main()
