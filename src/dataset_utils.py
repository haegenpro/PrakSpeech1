"""Canonical dataset import, integrity checks, and indexed feature storage."""
from dataclasses import asdict, dataclass
import csv
import hashlib
from importlib.metadata import version
import io
import json
import os
from pathlib import Path
import re
import tempfile

import numpy as np
import soundfile as sf

from .preprocessing import (DEFAULT_CONFIG, PreprocessingConfig, extract_mfcc39,
                            load_audio, load_feature_file, preprocess_audio)

VOWELS = ("a", "i", "u", "e", "o")
SPEAKERS = tuple(f"speaker{i:02d}" for i in range(1, 6))
PARTITIONS = {"template": Path("template"), "closed": Path("test/closed"),
              "open": Path("test/open")}
ALIASES = {"Peter": "speaker01", "S01": "speaker01", "Allen": "speaker02",
           "Haegen": "speaker03", "Nathan": "speaker04", "Nathaniel": "speaker04",
           "Indra": "speaker05"}


def digest(path: Path) -> str:
    """SHA256 of file bytes for provenance and cache validation."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_write(path: Path, content: bytes) -> None:
    """Replace a file only after a complete temporary write in the same folder."""
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            name = stream.name
            stream.write(content)
        os.replace(name, path)
    finally:
        if name and Path(name).exists():
            Path(name).unlink()


def write_json(path: Path, value: dict) -> None:
    atomic_write(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


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
    """Create folders and import legacy recordings without modifying originals.

    The persisted import manifest explicitly records each source-to-copy mapping.
    Existing destinations must have exactly the same bytes as their source.
    """
    root = Path(data_dir)
    for base in (root, Path(output_dir)):
        for partition, folder in PARTITIONS.items():
            speakers = SPEAKERS if partition != "open" else ("external_speaker01",)
            for speaker in speakers:
                target = base / folder / speaker
                target.mkdir(parents=True, exist_ok=True)
                (target / ".gitkeep").touch(exist_ok=True)
    mappings = []
    for path in sorted((root / "train").rglob("*.wav")):
        match = re.fullmatch(r"([A-Za-z0-9]+)_([AIUEO])_?(0?[12])\.wav", path.name)
        if not match or match[1] not in ALIASES:
            raise ValueError(f"Unmapped legacy recording: {path}")
        speaker, vowel, take = ALIASES[match[1]], match[2].lower(), int(match[3])
        partition = "template" if take == 1 else "closed"
        destination = root / PARTITIONS[partition] / speaker / f"{speaker}_{vowel}_{take:02d}.wav"
        mappings.append((path, destination))
    for vowel in VOWELS:
        for path in sorted((root / "test" / vowel).glob("*.wav")):
            match = re.fullmatch(r"G07_S(0[1-5])_([AIUEO])\.wav", path.name)
            if not match or match[2].lower() != vowel:
                raise ValueError(f"Unmapped external recording: {path}")
            speaker = f"external_g07_s{match[1]}"
            mappings.append((path, root / PARTITIONS["open"] / speaker / f"{speaker}_{vowel}_01.wav"))
    destinations = set()
    entries = []
    for source, destination in mappings:
        if destination in destinations:
            raise ValueError(f"Multiple sources assigned to {destination}")
        destinations.add(destination)
        source_hash = digest(source)
        if destination.exists() and digest(destination) != source_hash:
            raise ValueError(f"Conflicting destination: {destination}")
        entries.append({"source_path": source.relative_to(root).as_posix(),
                        "canonical_path": destination.relative_to(root).as_posix(),
                        "sha256": source_hash})
    for source, destination in mappings:
        if not destination.exists():
            atomic_write(destination, source.read_bytes())
    manifest = {"aliases": ALIASES, "recordings": entries}
    write_json(root / "import_manifest.json", manifest)
    return {"imported_recordings": len(entries)}


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
    missing = sorted(str(p) for p in expected - actual)
    return {"expected_internal": 50, "actual_internal": len(actual),
            "open_recordings": sum(r.partition == "open" for r in records),
            "decoded_recordings": decoded, "missing": missing, "errors": errors,
            "duplicate_audio": duplicates, "audio_formats": formats,
            "format_inconsistencies": len({tuple(v.values()) for v in formats.values()}) > 1,
            "valid": not (missing or errors or duplicates)}


def _signature(config: PreprocessingConfig) -> dict:
    return {"pipeline_version": 1, "config": asdict(config),
            "dependencies": {name: version(name) for name in
                             ("numpy", "scipy", "soundfile", "python_speech_features")}}


def extract_dataset_features(data_dir: str | Path = "data", output_dir: str | Path = "features",
                             config: PreprocessingConfig = DEFAULT_CONFIG,
                             overwrite: bool = False) -> dict:
    """Extract available canonical WAVs and replace the authoritative feature index.

    Invalid recordings are indexed with an error and cannot be loaded downstream.
    Existing files need matching provenance, configuration and content hashes.
    """
    root, output = Path(data_dir).resolve(), Path(output_dir).resolve()
    records, errors = discover_dataset(root)
    manifest_path = output / "manifest.json"
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    signature = _signature(config)
    provenance_path = root / "import_manifest.json"
    provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
    sources = {e["canonical_path"]: e["source_path"] for e in provenance.get("recordings", [])}
    rows, entries = [], {}
    summary = {"processed": 0, "reused": 0, "failed": len(errors), "errors": errors}
    for recording in records:
        relative = recording.path.relative_to(root).as_posix()
        feature_relative = str(Path(relative).with_suffix(".npy").as_posix())
        target = output / feature_relative
        row = dict(file_path=str(recording.path), filename=recording.path.name,
                   source_path=str(root / sources.get(relative, relative)),
                   speaker=recording.speaker, vowel=recording.vowel,
                   partition=recording.partition, sample_rate="", duration="",
                   number_of_samples="", original_sample_rate="", original_channels="",
                   original_samples="", original_duration="", number_of_frames="",
                   feature_dimensions="", feature_path=feature_relative,
                   validation_status="error", error="")
        try:
            source_hash = digest(recording.path)
            previous = old.get("recordings", {}).get(relative, {})
            compatible = (old.get("signature") == signature and
                          previous.get("source_sha256") == source_hash and
                          previous.get("feature_path") == feature_relative and target.exists()
                          and previous.get("feature_sha256") == digest(target))
            if target.exists() and not compatible and not overwrite:
                raise ValueError("Incompatible existing feature; rerun with --overwrite")
            info = sf.info(recording.path)
            audio, rate = load_audio(recording.path)
            audio, rate = preprocess_audio(audio, rate, config)
            if compatible:
                features = load_feature_file(target)
                summary["reused"] += 1
            else:
                features = extract_mfcc39(audio, rate, config)
                buffer = io.BytesIO()
                np.save(buffer, features, allow_pickle=False)
                atomic_write(target, buffer.getvalue())
                summary["processed"] += 1
            row.update(sample_rate=rate, duration=len(audio) / rate,
                       number_of_samples=len(audio), original_sample_rate=info.samplerate,
                       original_channels=info.channels, original_samples=info.frames,
                       original_duration=info.duration, number_of_frames=len(features),
                       feature_dimensions=39, validation_status="ok")
            entries[relative] = {"source_sha256": source_hash, "feature_path": feature_relative,
                                 "feature_sha256": digest(target)}
        except (ValueError, OSError, RuntimeError) as exc:
            row["error"] = str(exc)
            summary["failed"] += 1
            summary["errors"].append(f"{relative}: {exc}")
        rows.append(row)
    fields = (list(rows[0]) if rows else ["file_path", "filename", "source_path", "speaker",
              "vowel", "partition", "sample_rate", "duration", "number_of_samples",
              "original_sample_rate", "original_channels", "original_samples", "original_duration",
              "number_of_frames", "feature_dimensions", "feature_path", "validation_status", "error"])
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    atomic_write(output / "metadata.csv", stream.getvalue().encode("utf-8"))
    write_json(manifest_path, {"signature": signature, "data_root": str(root), "recordings": entries})
    return summary


def load_dataset(partition: str, features_dir: str | Path = "features") -> list[dict]:
    """Load current indexed sequences; reject changed audio or feature contents.

    Returns dictionaries with features, vowel, speaker and canonical WAV file_path.
    Original WAVs must remain available so provenance can be checked.
    """
    if partition not in PARTITIONS:
        raise ValueError(f"Unknown partition: {partition}")
    output = Path(features_dir).resolve()
    manifest = json.loads((output / "manifest.json").read_text())
    root = Path(manifest["data_root"])
    result = []
    with (output / "metadata.csv").open(newline="", encoding="utf-8") as stream:
        rows = sorted(csv.DictReader(stream), key=lambda row: row["file_path"])
    for row in rows:
        if row["partition"] != partition or row["validation_status"] != "ok":
            continue
        source = Path(row["file_path"])
        recording = parse_recording(source, root)
        key = source.relative_to(root).as_posix()
        entry = manifest["recordings"].get(key)
        target = (output / row["feature_path"]).resolve()
        if (not target.is_relative_to(output) or not entry
                or entry["feature_path"] != row["feature_path"]
                or not source.exists() or digest(source) != entry["source_sha256"]
                or not target.exists() or digest(target) != entry["feature_sha256"]):
            raise ValueError(f"Stale or incompatible feature index for {source}; re-extract")
        if (recording.partition != partition or recording.speaker != row["speaker"]
                or recording.vowel != row["vowel"]):
            raise ValueError(f"Metadata identity mismatch for {source}")
        result.append({"features": load_feature_file(target), "vowel": row["vowel"],
                       "speaker": row["speaker"], "file_path": str(source)})
    return result
