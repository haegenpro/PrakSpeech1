"""Dataset import, explicit validation, and metadata-based feature loading."""
from dataclasses import dataclass
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil

import numpy as np
import soundfile as sf

from .preprocessing import (DEFAULT_CONFIG, PreprocessingConfig, extract_mfcc39,
                            load_audio, load_feature_file, preprocess_audio)

VOWELS = ("a", "i", "u", "e", "o")
SPEAKERS = tuple(f"speaker{i:02d}" for i in range(1, 6))
PARTITIONS = {"template": Path("template"), "closed": Path("test/closed_speakers"),
              "open": Path("test/open_speakers")}
ALIASES = {"Peter": "speaker01", "S01": "speaker01", "Allen": "speaker02",
           "Haegen": "speaker03", "Nathan": "speaker04", "Nathaniel": "speaker04",
           "Indra": "speaker05"}


def write_json(path: Path, value: dict) -> None:
    """Save an explicit validation report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class Recording:
    path: Path
    speaker: str
    vowel: str
    partition: str
    take: str


def parse_recording(path: Path, data_dir: Path) -> Recording:
    """Validate canonical directory, filename, label, speaker and take together."""
    relative = path.relative_to(data_dir)
    for partition, folder in PARTITIONS.items():
        try:
            remainder = relative.relative_to(folder)
        except ValueError:
            continue
        if len(remainder.parts) != 2:
            break
        speaker = remainder.parts[0]
        match = re.fullmatch(r"(.+)_([aiueo])_(\d{2})\.wav", path.name)
        if not match or match[1] != speaker:
            break
        if partition != "open" and (speaker not in SPEAKERS or match[3] !=
                                     ("01" if partition == "template" else "02")):
            break
        if partition == "open" and (speaker in SPEAKERS or
                                    not re.fullmatch(r"external_[a-zA-Z0-9_]+", speaker)):
            break
        return Recording(path, speaker, match[2], partition, match[3])
    raise ValueError(f"Invalid canonical recording path or identity: {relative}")


def discover_dataset(data_dir: str | Path = "data") -> tuple[list[Recording], list[str]]:
    """Discover canonical WAVs only; legacy source directories are not partitions."""
    root = Path(data_dir)
    records, errors = [], []
    for folder in PARTITIONS.values():
        for path in sorted((root / folder).rglob("*")):
            if not path.is_file() or path.suffix.lower() != ".wav":
                continue
            try:
                records.append(parse_recording(path, root))
            except ValueError as exc:
                errors.append(str(exc))
    return records, errors


def setup_dataset(data_dir: str | Path = "data", output_dir: str | Path = "features") -> dict:
    """Copy legacy recordings after checking all mappings; preserve originals."""
    root = Path(data_dir).resolve()
    def canonical_relative(value):
        value = Path(value).as_posix()
        for old, new in (("test/closed/", "test/closed_speakers/"),
                         ("test/open/", "test/open_speakers/")):
            if value.startswith(old):
                return new + value[len(old):]
        return value

    def safe_path(relative):
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Path escapes dataset: {relative}")
        return path

    mappings = []
    for path in sorted((root / "train").rglob("*.wav")):
        match = re.fullmatch(r"([A-Za-z0-9]+)_([AIUEO])_?(0?[12])\.wav", path.name)
        if (not match or match[1] not in ALIASES
                or path.parent != root / "train" / match[2].lower()):
            raise ValueError(f"Unmapped legacy recording: {path}")
        speaker, vowel, take = ALIASES[match[1]], match[2].lower(), int(match[3])
        partition = "template" if take == 1 else "closed"
        destination = root / PARTITIONS[partition] / speaker / f"{speaker}_{vowel}_{take:02d}.wav"
        mappings.append((path, destination))
    for vowel in VOWELS:
        for path in sorted((root / "test" / vowel).rglob("*.wav")):
            match = re.fullmatch(r"(?:G(\d{2})|Kel(\d+))_S(\d{2})_([AIUEO])(\d{2})?\.wav", path.name)
            if not match or match[4].lower() != vowel or path.parent != root / "test" / vowel:
                raise ValueError(f"Unmapped external recording: {path}")
            group = int(match[1] or match[2])
            speaker = f"external_g{group:02d}_s{match[3]}"
            take = match[5] or "01"
            mappings.append((path, root / PARTITIONS["open"] / speaker / f"{speaker}_{vowel}_{take}.wav"))
    for old in ("test/closed", "test/open"):
        for source in sorted((root / old).rglob("*.wav")):
            destination = safe_path(canonical_relative(source.relative_to(root)))
            parse_recording(destination, root)
            mappings.append((source, destination))

    candidates = {}
    for source, destination in mappings:
        safe_path(source.relative_to(root))
        safe_path(destination.relative_to(root))
        parse_recording(destination, root)
        content = source.read_bytes()
        if destination in candidates and candidates[destination].read_bytes() != content:
            raise ValueError(f"Conflicting sources for {destination}")
        if destination.exists() and destination.read_bytes() != content:
            raise ValueError(f"Conflicting destination: {destination}")
        candidates[destination] = source
    _, errors = discover_dataset(root)
    if errors:
        raise ValueError("; ".join(errors))
    imported = 0
    for destination, source in candidates.items():
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            imported += 1
    for base in (root, Path(output_dir).resolve()):
        for partition, folder in PARTITIONS.items():
            (base / folder).mkdir(parents=True, exist_ok=True)
            if partition != "open":
                for speaker in SPEAKERS:
                    (base / folder / speaker).mkdir(parents=True, exist_ok=True)
    return {"imported_recordings": imported}



def validate_dataset(data_dir: str | Path = "data") -> dict:
    """Report completeness, decoded content duplicates, errors and format differences."""
    root = Path(data_dir)
    records, errors = discover_dataset(root)
    expected = {root / PARTITIONS[p] / s / f"{s}_{v}_{t}.wav"
                for p, t in (("template", "01"), ("closed", "02"))
                for s in SPEAKERS for v in VOWELS}
    actual = {r.path for r in records if r.partition != "open"}
    hashes, formats, duplicates = {}, {}, []
    decoded = 0
    for recording in records:
        try:
            audio, rate = load_audio(recording.path)
            info = sf.info(recording.path)
            formats[str(recording.path)] = {"sample_rate": rate, "channels": info.channels,
                                             "subtype": info.subtype}
            key = hashlib.sha256(str(rate).encode() + audio.tobytes()).hexdigest()
            if key in hashes:
                duplicates.append([hashes[key], str(recording.path)])
            hashes[key] = str(recording.path)
            decoded += 1
        except (ValueError, OSError) as exc:
            errors.append(f"{recording.path}: {exc}")
    open_speakers = sorted({r.speaker for r in records if r.partition == "open"})
    open_missing = [f"{speaker}/{vowel}" for speaker in open_speakers for vowel in VOWELS
                    if not any(r.speaker == speaker and r.vowel == vowel for r in records
                               if r.partition == "open")]
    missing = sorted(str(p) for p in expected - actual)
    return {"expected_internal": 50, "actual_internal": len(actual),
            "open_recordings": sum(r.partition == "open" for r in records),
            "open_speakers": len(open_speakers), "missing_open_vowels": open_missing,
            "decoded_recordings": decoded, "missing": missing, "errors": errors,
            "duplicate_audio": duplicates, "audio_formats": formats,
            "format_inconsistencies": len({tuple(v.values()) for v in formats.values()}) > 1,
            "valid": not (missing or errors or duplicates or open_missing)}


def extract_dataset_features(data_dir: str | Path = "data", output_dir: str | Path = "features",
                             config: PreprocessingConfig = DEFAULT_CONFIG) -> dict:
    """Regenerate all canonical features and replace the metadata index.

    Failed recordings receive error rows and are excluded from downstream loading.
    """
    root, output = Path(data_dir).resolve(), Path(output_dir).resolve()
    records, errors = discover_dataset(root)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    summary = {"processed": 0, "failed": len(errors), "errors": errors}
    for recording in records:
        relative = recording.path.relative_to(root).as_posix()
        feature_relative = str(Path(relative).with_suffix(".npy").as_posix())
        target = output / feature_relative
        row = dict(file_path=relative, filename=recording.path.name,
                   source_path=relative,
                   speaker=recording.speaker, vowel=recording.vowel,
                   partition=recording.partition, sample_rate="", duration="",
                   number_of_samples="", original_sample_rate="", original_channels="",
                   original_samples="", original_duration="", number_of_frames="",
                   feature_dimensions="", feature_path=feature_relative,
                   validation_status="error", error="")
        try:
            info = sf.info(recording.path)
            audio, rate = load_audio(recording.path)
            audio, rate = preprocess_audio(audio, rate, config)
            features = extract_mfcc39(audio, rate, config)
            target.parent.mkdir(parents=True, exist_ok=True)
            np.save(target, features, allow_pickle=False)
            summary["processed"] += 1
            row.update(sample_rate=rate, duration=len(audio) / rate,
                       number_of_samples=len(audio), original_sample_rate=info.samplerate,
                       original_channels=info.channels, original_samples=info.frames,
                       original_duration=info.duration, number_of_frames=len(features),
                       feature_dimensions=39, validation_status="ok")
        except (ValueError, OSError, RuntimeError) as exc:
            row["error"] = str(exc)
            summary["failed"] += 1
            summary["errors"].append(f"{relative}: {exc}")
        rows.append(row)
    fields = (list(rows[0]) if rows else ["file_path", "filename", "source_path", "speaker",
              "vowel", "partition", "sample_rate", "duration", "number_of_samples",
              "original_sample_rate", "original_channels", "original_samples", "original_duration",
              "number_of_frames", "feature_dimensions", "feature_path", "validation_status", "error"])
    with (output / "metadata.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return summary


def load_dataset(partition: str, features_dir: str | Path = "features") -> list[dict]:
    """Load validated indexed features without requiring the original WAVs.

    WAV identifiers are resolved against the sibling data directory.
    """
    if partition not in PARTITIONS:
        raise ValueError(f"Unknown partition: {partition}")
    output = Path(features_dir).resolve()
    root = output.parent / "data"
    result = []
    with (output / "metadata.csv").open(newline="", encoding="utf-8") as stream:
        rows = sorted(csv.DictReader(stream), key=lambda row: row["file_path"])
    for row in rows:
        if row["partition"] != partition or row["validation_status"] != "ok":
            continue
        source_relative = Path(row["file_path"])
        feature_relative = Path(row["feature_path"])
        source = (root / source_relative).resolve()
        target = (output / feature_relative).resolve()
        if (source_relative.is_absolute() or feature_relative.is_absolute()
                or not source.is_relative_to(root) or not target.is_relative_to(output)):
            raise ValueError(f"Metadata path escapes dataset or features: {row['file_path']}")
        recording = parse_recording(source, root)
        if feature_relative != source_relative.with_suffix(".npy"):
            raise ValueError(f"Metadata feature path mismatch for {source}")
        if not target.is_file():
            raise ValueError(f"Missing feature file: {target}; rerun extraction")
        if (recording.partition != partition or recording.speaker != row["speaker"]
                or recording.vowel != row["vowel"]):
            raise ValueError(f"Metadata identity mismatch for {source}")
        result.append({"features": load_feature_file(target), "vowel": row["vowel"],
                       "speaker": row["speaker"], "file_path": str(source)})
    return result
