"""Dataset isolation, import collisions, regeneration, and feature loading."""
import csv
from dataclasses import replace
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf

from src.dataset_utils import (ALIASES, VOWELS, discover_dataset, extract_dataset_features,
                               load_dataset, setup_dataset, validate_dataset)
from src.preprocessing import DEFAULT_CONFIG


def write_tone(path, frequency=440):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, 0.3 * np.sin(2 * np.pi * frequency * np.arange(3200) / 16000), 16000)
    return path


def test_indexing_separation(tmp_path):
    root = tmp_path / "data"
    for folder, speaker, take in [("template", "speaker01", "01"),
                                  ("test/closed_speakers", "speaker01", "02"),
                                  ("test/open_speakers", "external_g07_s01", "01")]:
        write_tone(root / folder / speaker / f"{speaker}_a_{take}.wav")
    records, errors = discover_dataset(root)
    assert not errors
    assert {r.partition for r in records} == {"template", "closed", "open"}
    assert {r.speaker for r in records if r.partition == "open"}.isdisjoint(
        r.speaker for r in records if r.partition == "template")


def test_missing_invalid_and_duplicate(tmp_path):
    root = tmp_path / "data"
    write_tone(root / "template/speaker01/speaker01_a_01.wav")
    write_tone(root / "test/closed_speakers/speaker01/speaker01_a_02.wav")
    write_tone(root / "test/open_speakers/speaker01/speaker01_a_01.wav")
    write_tone(root / "template/speaker01/wrong.wav")
    report = validate_dataset(root)
    assert len(report["missing"]) == 48
    assert len(report["errors"]) == 2 and len(report["duplicate_audio"]) == 1
    assert not report["valid"]


def test_open_speakers_discovered_and_missing_vowels(tmp_path):
    root = tmp_path / "data"
    for group in (3, 4, 7, 9, 12):
        speaker = f"external_g{group:02d}_s01"
        for i, vowel in enumerate(VOWELS):
            if group == 12 and vowel == "o":
                continue
            write_tone(root / "test/open_speakers" / speaker / f"{speaker}_{vowel}_01.wav",
                       200 + group * 30 + i)
    report = validate_dataset(root)
    assert report["open_speakers"] == 5 and report["open_recordings"] == 24
    assert report["missing_open_vowels"] == ["external_g12_s01/o"]


def test_import_aliases_idempotent_and_conflict(tmp_path):
    for i, alias in enumerate(ALIASES):
        root = tmp_path / alias
        source = write_tone(root / "train/a" / f"{alias}_A1.wav", 300 + i * 30)
        output = root / "features"
        assert setup_dataset(root, output)["imported_recordings"] == 1
        destination = root / "template" / ALIASES[alias] / f"{ALIASES[alias]}_a_01.wav"
        assert destination.read_bytes() == source.read_bytes()
        assert setup_dataset(root, output)["imported_recordings"] == 0
        assert not (root / "import_manifest.json").exists()
        write_tone(destination, 900)
        with pytest.raises(ValueError, match="Conflicting"):
            setup_dataset(root, output)
        assert source.exists()


@pytest.mark.parametrize("name,speaker", [
    ("Kel3_S01_A.wav", "external_g03_s01"),
    ("G04_S03_A.wav", "external_g04_s03"),
    ("G07_S01_A.wav", "external_g07_s01"),
    ("G09_S01_A01.wav", "external_g09_s01"),
])
def test_external_naming_conventions(tmp_path, name, speaker):
    root = tmp_path / "data"
    source = write_tone(root / "test/a" / name)
    setup_dataset(root, tmp_path / "features")
    target = root / "test/open_speakers" / speaker / f"{speaker}_a_01.wav"
    assert target.read_bytes() == source.read_bytes()
    assert setup_dataset(root, tmp_path / "features")["imported_recordings"] == 0


def test_group_identity_and_legacy_canonical_copy(tmp_path):
    root = tmp_path / "data"
    for group in ("G04", "G07", "G09", "Kel3"):
        write_tone(root / "test/a" / f"{group}_S03_A.wav")
    legacy = write_tone(root / "test/open/external_g07_s03/external_g07_s03_a_01.wav")
    closed = write_tone(root / "test/closed/speaker01/speaker01_a_02.wav")
    setup_dataset(root, tmp_path / "features")
    records, errors = discover_dataset(root)
    assert not errors
    assert len({r.speaker for r in records if r.partition == "open"}) == 4
    assert legacy.exists() and closed.exists()


@pytest.mark.parametrize("folder,name", [("test/a", "G04_S03_E.wav"),
                                         ("test/a", "unknown.wav"),
                                         ("train/e", "Indra_A1.wav")])
def test_invalid_mapping_prevents_all_copies(tmp_path, folder, name):
    root = tmp_path / "data"
    good = write_tone(root / "test/a/G07_S01_A.wav")
    bad = write_tone(root / folder / name)
    with pytest.raises(ValueError, match="Unmapped"):
        setup_dataset(root, tmp_path / "features")
    assert good.exists() and bad.exists()
    assert not (root / "test/open_speakers").exists()


def test_conflicting_sources_preflight(tmp_path):
    root = tmp_path / "data"
    first = write_tone(root / "test/a/Kel3_S01_A.wav")
    second = write_tone(root / "test/a/G03_S01_A.wav", 800)
    with pytest.raises(ValueError, match="Conflicting"):
        setup_dataset(root, tmp_path / "features")
    assert first.exists() and second.exists()
    assert not (root / "test/open_speakers").exists()


def test_regeneration_updates_audio_and_configuration(tmp_path):
    root, output = tmp_path / "data", tmp_path / "features"
    source = write_tone(root / "template/speaker01/speaker01_a_01.wav")
    assert extract_dataset_features(root, output)["processed"] == 1
    before = load_dataset("template", output)[0]["features"].copy()
    target = output / "template/speaker01/speaker01_a_01.npy"
    target.write_bytes(b"broken")
    assert extract_dataset_features(root, output)["processed"] == 1
    np.testing.assert_array_equal(load_dataset("template", output)[0]["features"], before)
    write_tone(source, 800)
    assert extract_dataset_features(root, output)["processed"] == 1
    assert not np.array_equal(load_dataset("template", output)[0]["features"], before)
    assert extract_dataset_features(root, output, replace(DEFAULT_CONFIG, normalize=True))["failed"] == 0
    assert not (output / "manifest.json").exists()
    assert load_dataset("open", output) == []


def test_loading_without_sources_and_manifests(tmp_path):
    root, output = tmp_path / "data", tmp_path / "custom_features"
    source = write_tone(root / "template/speaker01/speaker01_a_01.wav")
    extract_dataset_features(root, output)
    expected = load_dataset("template", output)[0]["features"].copy()
    source.unlink()
    loaded = load_dataset("template", output)[0]
    np.testing.assert_array_equal(loaded["features"], expected)
    assert loaded["file_path"] == str(source.resolve())
    np.save(output / "unindexed.npy", np.zeros((1, 39)))
    assert len(load_dataset("template", output)) == 1
    extract_dataset_features(root, output)
    assert load_dataset("template", output) == []


@pytest.mark.parametrize("change", ["missing", "shape", "nonfinite", "identity", "escape", "source_escape", "wrong_feature"])
def test_invalid_indexed_features_rejected(tmp_path, change):
    root, output = tmp_path / "data", tmp_path / "features"
    write_tone(root / "template/speaker01/speaker01_a_01.wav")
    extract_dataset_features(root, output)
    target = output / "template/speaker01/speaker01_a_01.npy"
    if change == "missing":
        target.unlink()
    elif change in ("shape", "nonfinite"):
        np.save(target, np.zeros((2, 13)) if change == "shape" else np.full((2, 39), np.nan))
    else:
        metadata = output / "metadata.csv"
        with metadata.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if change == "identity":
            rows[0]["speaker"] = "speaker02"
        elif change == "escape":
            rows[0]["feature_path"] = "../escape.npy"
        elif change == "source_escape":
            rows[0]["file_path"] = "../escape.wav"
        else:
            rows[0]["feature_path"] = "wrong.npy"
        with metadata.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    with pytest.raises(ValueError):
        load_dataset("template", output)


def test_corrupt_recording_is_reported_and_cli_fails(tmp_path):
    root, output = tmp_path / "data", tmp_path / "features"
    source = write_tone(root / "template/speaker01/speaker01_a_01.wav")
    extract_dataset_features(root, output)
    source.write_bytes(b"broken")
    summary = extract_dataset_features(root, output)
    assert summary["failed"] == 1 and summary["processed"] == 0
    assert load_dataset("template", output) == []
    result = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "scripts/extract_features.py"),
                             "--data-dir", str(root), "--output-dir", str(output)], capture_output=True)
    assert result.returncode == 1
