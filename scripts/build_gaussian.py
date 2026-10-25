import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset_utils import load_dataset
from src.gaussian import build_models, save_models


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", default="features")
    parser.add_argument("--output", default="models/gaussian_models.npz")
    args = parser.parse_args()
    models = build_models(load_dataset("template", args.features_dir))
    save_models(models, args.output)
    for vowel, model in models.items():
        print(f"{vowel}: mean {model['mean'].shape}, cov {model['cov'].shape}")


if __name__ == "__main__":
    main()
