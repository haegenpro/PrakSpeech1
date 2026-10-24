import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.gaussian import load_models, predict
from src.preprocessing import process_audio_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav")
    parser.add_argument("--models", default="models/gaussian_models.npz")
    args = parser.parse_args()
    vowel, costs = predict(process_audio_file(args.wav), load_models(args.models))
    for name, cost in costs.items():
        print(f"{name}: {cost:.3f}")
    print(f"prediction: {vowel}")


if __name__ == "__main__":
    main()
