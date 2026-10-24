"""Extract indexed 39-dimensional frame sequences for all available WAVs."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset_utils import extract_dataset_features
from src.preprocessing import PreprocessingConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="features")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--frame-length", type=float, default=0.025)
    parser.add_argument("--frame-step", type=float, default=0.01)
    parser.add_argument("--nfilt", type=int, default=26)
    parser.add_argument("--nfft", type=int, default=512)
    parser.add_argument("--preemphasis", type=float, default=0.97)
    parser.add_argument("--delta-window", type=int, default=2)
    parser.add_argument("--trim-silence", action="store_true")
    parser.add_argument("--trim-threshold", type=float, default=0.01)
    parser.add_argument("--normalize", action="store_true")
    args = parser.parse_args()
    config = PreprocessingConfig(target_sample_rate=args.sample_rate,
        frame_length=args.frame_length, frame_step=args.frame_step, nfilt=args.nfilt,
        nfft=args.nfft, preemphasis=args.preemphasis, delta_window=args.delta_window,
        trim_silence=args.trim_silence, trim_threshold=args.trim_threshold,
        normalize=args.normalize)
    summary = extract_dataset_features(args.data_dir, args.output_dir, config, args.overwrite)
    print(json.dumps(summary, indent=2))
    sys.exit(1 if summary["failed"] else 0)


if __name__ == "__main__":
    main()
