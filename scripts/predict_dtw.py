import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dtw_predict import predict
from src.dtw_templates import load_templates
from src.preprocessing import process_audio_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav")
    parser.add_argument("--templates", default="models/dtw_templates.npz")
    args = parser.parse_args()
    vowel, costs = predict(process_audio_file(args.wav), load_templates(args.templates))
    for name, cost in costs.items():
        print(f"{name}: {cost:.3f}")
    print(f"prediction: {vowel}")


if __name__ == "__main__":
    main()
