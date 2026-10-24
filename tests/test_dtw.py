import numpy as np

from src.dtw import dtw
from src.dtw_predict import predict
from src.dtw_templates import average_template, build_templates, load_templates, save_templates
from src.silence import trim_silence


def sequence(length, offset=0.0, seed=0):
    return np.random.default_rng(seed).normal(offset, 1, (length, 39))


def test_identical_sequences_cost_zero():
    a = sequence(20)
    cost, path = dtw(a, a)
    assert cost == 0 and path == [(i, i) for i in range(20)]


def test_path_is_monotonic_and_complete():
    cost, path = dtw(sequence(15), sequence(30, seed=1))
    steps = np.diff(path, axis=0)
    assert path[0] == (0, 0) and path[-1] == (14, 29) and cost > 0
    assert ((steps >= 0) & (steps <= 1)).all() and (steps.sum(axis=1) > 0).all()


def test_time_stretch_costs_nothing():
    a = sequence(20)
    assert dtw(a, np.repeat(a, 2, axis=0))[0] == 0 < dtw(a, sequence(20, 3, seed=2))[0]


def test_trim_silence_keeps_loud_frames():
    features = sequence(30)
    features[:, 0] = -12
    features[10:20, 0] = -1
    assert np.array_equal(trim_silence(features), features[10:20])


def test_average_of_copies_is_unchanged():
    a = sequence(20)
    assert np.allclose(average_template([a, a, np.repeat(a, 2, axis=0)]), a)


def test_templates_round_trip_and_predict(tmp_path):
    records = [{"vowel": vowel, "features": sequence(25 + speaker, offset, seed=speaker)}
               for vowel, offset in (("a", 0), ("i", 5)) for speaker in range(3)]
    save_templates(build_templates(records), tmp_path / "templates.npz")
    templates = load_templates(tmp_path / "templates.npz")
    assert sorted(templates) == ["a", "i"]
    assert all(template.shape[1] == 39 for template in templates.values())
    assert predict(sequence(28, 5, seed=9), templates)[0] == "i"
