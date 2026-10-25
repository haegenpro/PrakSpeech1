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
PARTITIONS = {"template": Path("template"), "closed": Path("test/closed_speakers"),
              "open": Path("test/open_speakers")}
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
    """Consolidate legacy WAVs after preflight and byte-for-byte verification.

    Historical source paths remain in the manifest after duplicate originals are
    removed. A second run verifies canonical files without losing provenance.
    """
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
    legacy_dirs = [root / "train", *(root / "test" / v for v in VOWELS)]
    for old in ("test/closed", "test/open"):
        legacy_dirs.append(root / old)
        for source in sorted((root / old).rglob("*.wav")):
            destination = safe_path(canonical_relative(source.relative_to(root)))
            parse_recording(destination, root)
            mappings.append((source, destination))

    manifest_path = root / "import_manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    entries = {}
    expected_hashes = {}
    for item in previous.get("recordings", []):
        entry = dict(item, canonical_path=canonical_relative(item["canonical_path"]))
        safe_path(entry["source_path"])
        destination = safe_path(entry["canonical_path"])
        parse_recording(destination, root)
        entries[entry["source_path"]] = entry
        if destination in expected_hashes and expected_hashes[destination] != entry["sha256"]:
            raise ValueError(f"Conflicting historical destination: {destination}")
        expected_hashes[destination] = entry["sha256"]
    candidates = {}
    newly_imported = set()
    for source, destination in mappings:
        safe_path(source.relative_to(root))
        if not source.resolve().is_relative_to(root):
            raise ValueError(f"Source escapes dataset: {source}")
        source_hash = digest(source)
        if destination in expected_hashes and expected_hashes[destination] != source_hash:
            raise ValueError(f"Conflicting destination: {destination}")
        if destination.exists() and digest(destination) != source_hash:
            raise ValueError(f"Conflicting destination: {destination}")
        expected_hashes[destination] = source_hash
        candidates[destination] = source
        source_relative = source.relative_to(root).as_posix()
        # Preserve upload provenance rather than adding redundant migration aliases.
        if not source_relative.startswith(("test/closed/", "test/open/")) or not any(
                e["canonical_path"] == destination.relative_to(root).as_posix() for e in entries.values()):
            entries[source_relative] = {"source_path": source_relative,
                "canonical_path": destination.relative_to(root).as_posix(), "sha256": source_hash}
            newly_imported.add(destination)
    for destination, expected in expected_hashes.items():
        if destination.exists():
            if digest(destination) != expected:
                raise ValueError(f"Conflicting destination: {destination}")
        elif destination not in candidates:
            raise ValueError(f"Missing canonical recording: {destination}")
    records, errors = discover_dataset(root)
    if errors:
        raise ValueError("; ".join(errors))
    for destination, source in candidates.items():
        if not destination.exists():
            atomic_write(destination, source.read_bytes())
    for destination, expected in expected_hashes.items():
        if digest(destination) != expected:
            raise ValueError(f"Verification failed: {destination}")
    # Recheck every original before removing any source, including duplicate aliases.
    for source, destination in mappings:
        if digest(source) != digest(destination):
            raise ValueError(f"Source changed during import: {source}")
    manifest = {**previous, "aliases": {**ALIASES, **previous.get("aliases", {})},
                "recordings": sorted(entries.values(), key=lambda e: e["source_path"])}
    write_json(manifest_path, manifest)
    for source, destination in mappings:
        source.unlink()
    for folder in legacy_dirs:
        if folder.exists():
            for directory in sorted([folder, *(p for p in folder.rglob("*") if p.is_dir())],
                                    key=lambda p: len(p.parts), reverse=True):
                placeholder = directory / ".gitkeep"
                if placeholder.is_file():
                    placeholder.unlink()
                if not any(directory.iterdir()):
                    directory.rmdir()
    for base in (root, Path(output_dir).resolve()):
        placeholder_dir = base / PARTITIONS["open"] / "external_speaker01"
        if placeholder_dir.exists() and all(p.name == ".gitkeep" for p in placeholder_dir.iterdir()):
            (placeholder_dir / ".gitkeep").unlink(missing_ok=True)
            placeholder_dir.rmdir()
        for partition, folder in PARTITIONS.items():
            (base / folder).mkdir(parents=True, exist_ok=True)
            if partition != "open":
                for speaker in SPEAKERS:
                    (base / folder / speaker).mkdir(parents=True, exist_ok=True)
    return {"imported_recordings": len(newly_imported), "removed_duplicates": len(mappings),
            "provenance_entries": len(entries)}


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
        row = dict(file_path=relative, filename=recording.path.name,
                   source_path=sources.get(relative, relative),
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
    data_root = Path(os.path.relpath(root, output)).as_posix()
    write_json(manifest_path, {"signature": signature, "data_root": data_root, "recordings": entries})
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
    root = (output / manifest["data_root"]).resolve()
    result = []
    with (output / "metadata.csv").open(newline="", encoding="utf-8") as stream:
        rows = sorted(csv.DictReader(stream), key=lambda row: row["file_path"])
    for row in rows:
        if row["partition"] != partition or row["validation_status"] != "ok":
            continue
        source = root / row["file_path"]
        recording = parse_recording(source, root)
        entry = manifest["recordings"].get(row["file_path"])
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
