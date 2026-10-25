"""Dataset isolation, import provenance and cache-integrity checks."""
from dataclasses import replace
import json

import numpy as np
import pytest
import soundfile as sf

from src.dataset_utils import (ALIASES, discover_dataset, extract_dataset_features,
                               load_dataset, setup_dataset, validate_dataset)
from src.preprocessing import DEFAULT_CONFIG


def write_tone(path, frequency=440, duration=0.2):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, 0.3 * np.sin(2 * np.pi * frequency * np.arange(int(16000 * duration)) / 16000), 16000)
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
    assert {r.vowel for r in records} == {"a"}
    assert len({r.path for r in records}) == 3
    assert {r.speaker for r in records if r.partition == "open"}.isdisjoint(
        r.speaker for r in records if r.partition == "template")


def test_missing_invalid_and_duplicate(tmp_path):
    root = tmp_path / "data"
    write_tone(root / "template/speaker01/speaker01_a_01.wav")
    write_tone(root / "test/closed_speakers/speaker01/speaker01_a_02.wav")
    write_tone(root / "test/open_speakers/speaker01/speaker01_a_01.wav")
    write_tone(root / "template/speaker01/wrong.wav")
    report = validate_dataset(root)
    assert report["expected_internal"] == 50 and len(report["missing"]) == 48
    assert len(report["errors"]) == 2 and len(report["duplicate_audio"]) == 1
    assert not report["valid"]


def test_import_aliases_idempotent_and_conflict(tmp_path):
    root, output = tmp_path / "data", tmp_path / "features"
    for i, alias in enumerate(ALIASES):
        alias_root = tmp_path / alias
        source = write_tone(alias_root / "train/a" / f"{alias}_A1.wav", 300 + i * 30)
        # Test each alias independently to avoid intentionally colliding destinations.
        setup_dataset(alias_root, output)
        destination = alias_root / "template" / ALIASES[alias] / f"{ALIASES[alias]}_a_01.wav"
        assert destination.exists()
        assert not source.exists()
        manifest = (alias_root / "import_manifest.json").read_bytes()
        setup_dataset(alias_root, output)
        assert (alias_root / "import_manifest.json").read_bytes() == manifest
    source = write_tone(root / "train/e/S01_E_02.wav")
    write_tone(root / "test/e/G07_S01_E.wav", 600)
    assert setup_dataset(root, output)["imported_recordings"] == 2
    manifest = json.loads((root / "import_manifest.json").read_text())
    assert len(manifest["recordings"]) == 2
    assert not source.exists()
    write_tone(root / "test/closed_speakers/speaker01/speaker01_e_02.wav", 900)
    with pytest.raises(ValueError, match="Conflicting"):
        setup_dataset(root, output)


@pytest.mark.parametrize("name,speaker", [
    ("Kel3_S01_A.wav", "external_g03_s01"),
    ("G04_S03_A.wav", "external_g04_s03"),
    ("G07_S01_A.wav", "external_g07_s01"),
    ("G09_S01_A01.wav", "external_g09_s01"),
])
def test_external_naming_conventions(tmp_path, name, speaker):
    root = tmp_path / "data"
    source = write_tone(root / "test/a" / name)
    original = source.read_bytes()
    setup_dataset(root, tmp_path / "features")
    target = root / "test/open_speakers" / speaker / f"{speaker}_a_01.wav"
    assert target.read_bytes() == original
    assert not source.exists()
    before = (root / "import_manifest.json").read_bytes()
    setup_dataset(root, tmp_path / "features")
    assert (root / "import_manifest.json").read_bytes() == before


def test_group_identity_and_legacy_canonical_migration(tmp_path):
    root = tmp_path / "data"
    for group in ("G04", "G07", "G09", "Kel3"):
        write_tone(root / "test/a" / f"{group}_S03_A.wav")
    legacy = write_tone(root / "test/open/external_g07_s03/external_g07_s03_a_01.wav")
    write_tone(root / "test/closed/speaker01/speaker01_a_02.wav")
    setup_dataset(root, tmp_path / "features")
    records, errors = discover_dataset(root)
    assert not errors
    assert len({r.speaker for r in records if r.partition == "open"}) == 4
    assert not legacy.exists()
    assert not (root / "test/open").exists()
    assert not (root / "test/closed").exists()


@pytest.mark.parametrize("name", ["G04_S03_E.wav", "unknown.wav"])
def test_unmapped_or_mismatched_source_stops_cleanup(tmp_path, name):
    root = tmp_path / "data"
    good = write_tone(root / "test/a/G07_S01_A.wav")
    bad = write_tone(root / "test/a" / name)
    with pytest.raises(ValueError, match="Unmapped"):
        setup_dataset(root, tmp_path / "features")
    assert good.exists() and bad.exists()


def test_conflicting_sources_preserved_and_history_verified(tmp_path):
    root = tmp_path / "data"
    first = write_tone(root / "test/a/Kel3_S01_A.wav")
    second = write_tone(root / "test/a/G03_S01_A.wav", 800)
    with pytest.raises(ValueError, match="Conflicting"):
        setup_dataset(root, tmp_path / "features")
    assert first.exists() and second.exists()
    second.unlink()
    setup_dataset(root, tmp_path / "features")
    target = root / "test/open_speakers/external_g03_s01/external_g03_s01_a_01.wav"
    target.unlink()
    with pytest.raises(ValueError, match="Missing canonical"):
        setup_dataset(root, tmp_path / "features")


def test_historical_manifest_migration_preserves_provenance(tmp_path):
    from src.dataset_utils import digest

    root = tmp_path / "data"
    legacy = write_tone(root / "test/open/external_g07_s01/external_g07_s01_a_01.wav")
    entry = {"source_path": "test/a/G07_S01_A.wav",
             "canonical_path": legacy.relative_to(root).as_posix(), "sha256": digest(legacy)}
    (root / "import_manifest.json").write_text(json.dumps({"aliases": ALIASES, "recordings": [entry]}))
    note = root / "test/a/notes.txt"
    note.parent.mkdir(parents=True)
    note.write_text("Keep this upload note")
    setup_dataset(root, tmp_path / "features")
    manifest = json.loads((root / "import_manifest.json").read_text())
    assert manifest["recordings"] == [{**entry, "canonical_path":
        "test/open_speakers/external_g07_s01/external_g07_s01_a_01.wav"}]
    assert note.read_text() == "Keep this upload note"
    assert not legacy.exists()
    setup_dataset(root, tmp_path / "features")
    assert json.loads((root / "import_manifest.json").read_text()) == manifest


def test_internal_vowel_mismatch_preserves_all_sources(tmp_path):
    root = tmp_path / "data"
    good = write_tone(root / "train/a/Allen_A1.wav")
    bad = write_tone(root / "train/e/Indra_A1.wav")
    with pytest.raises(ValueError, match="Unmapped"):
        setup_dataset(root, tmp_path / "features")
    assert good.exists() and bad.exists()


def test_partial_extraction_reuse_and_incompatible_outputs(tmp_path):
    root, output = tmp_path / "data", tmp_path / "features"
    source = write_tone(root / "template/speaker01/speaker01_a_01.wav")
    summary = extract_dataset_features(root, output)
    assert summary["processed"] == 1 and summary["failed"] == 0
    data = load_dataset("template", output)
    assert len(data) == 1 and data[0]["features"].shape[1] == 39
    assert load_dataset("open", output) == []
    assert extract_dataset_features(root, output)["reused"] == 1
    changed = replace(DEFAULT_CONFIG, normalize=True)
    assert extract_dataset_features(root, output, changed)["failed"] == 1
    assert load_dataset("template", output) == []
    assert extract_dataset_features(root, output, changed, overwrite=True)["processed"] == 1
    write_tone(source, 800)
    with pytest.raises(ValueError, match="Stale"):
        load_dataset("template", output)


def test_feature_tampering_and_removed_source(tmp_path):
    root, output = tmp_path / "data", tmp_path / "features"
    source = write_tone(root / "template/speaker01/speaker01_a_01.wav")
    extract_dataset_features(root, output)
    target = output / "template/speaker01/speaker01_a_01.npy"
    np.save(target, np.zeros((3, 39), dtype=np.float32))
    with pytest.raises(ValueError, match="Stale"):
        load_dataset("template", output)
    extract_dataset_features(root, output, overwrite=True)
    source.unlink()
    with pytest.raises(ValueError, match="Stale"):
        load_dataset("template", output)
    extract_dataset_features(root, output)
    assert load_dataset("template", output) == []


def test_corrupt_recording_is_reported(tmp_path):
    root, output = tmp_path / "data", tmp_path / "features"
    source = write_tone(root / "template/speaker01/speaker01_a_01.wav")
    source.write_bytes(b"broken")
    summary = extract_dataset_features(root, output)
    assert summary["failed"] == 1 and summary["processed"] == 0
    assert load_dataset("template", output) == []
