import numpy as np
from scipy.stats import multivariate_normal

from src.gaussian import build_models, fit_gaussian, load_models, nll, predict, save_models


def frames(count, mean=0.0, seed=0):
    return np.random.default_rng(seed).normal(mean, 1, (count, 39))


def test_fit_recovers_mean_and_covariance():
    rng = np.random.default_rng(0)
    mean = rng.normal(0, 3, 4)
    a = rng.normal(0, 1, (4, 4))
    cov = a @ a.T + np.eye(4)
    x = rng.multivariate_normal(mean, cov, 200000)
    model = fit_gaussian(x, eps=0)
    assert np.allclose(model["mean"], mean, atol=0.05)
    assert np.allclose(model["cov"], cov, atol=0.1)


def test_nll_matches_scipy():
    x, y = frames(300, seed=2), frames(40, 0.5, seed=3)
    model = fit_gaussian(x)
    expected = -multivariate_normal(model["mean"], model["cov"]).logpdf(y).mean()
    assert np.isclose(nll(y, model), expected)


def test_fewer_frames_than_dimensions_does_not_crash():
    model = fit_gaussian(frames(5))
    assert np.isfinite(nll(frames(3, seed=1), model))


def test_predict_separated_classes():
    records = [{"vowel": v, "features": frames(60, offset, seed=s)}
               for v, offset in (("a", 0), ("i", 6), ("u", -6)) for s in range(3)]
    models = build_models(records, trim=False)
    vowel, costs = predict(frames(30, 6, seed=9), models, trim=False)
    assert vowel == "i" and sorted(costs) == ["a", "i", "u"]
    assert costs["i"] < costs["a"] and costs["i"] < costs["u"]


def test_trim_removes_silence_before_fitting():
    speech = frames(40, 5)
    speech[:, 0] = -1
    padded = np.concatenate([frames(60, seed=4) * 0 + np.r_[-12, np.zeros(38)], speech])
    models = build_models([{"vowel": "a", "features": padded}])
    assert np.allclose(models["a"]["mean"], speech.mean(axis=0))


def test_cost_is_per_frame_average():
    model = fit_gaussian(frames(300))
    x = frames(40, seed=5)
    assert np.isclose(nll(np.concatenate([x, x]), model), nll(x, model))


def test_save_load_round_trip(tmp_path):
    records = [{"vowel": v, "features": frames(60, offset, seed=s)}
               for v, offset in (("a", 0), ("i", 4)) for s in range(3)]
    models = build_models(records, trim=False)
    save_models(models, tmp_path / "gaussian.npz")
    loaded = load_models(tmp_path / "gaussian.npz")
    x = frames(25, 4, seed=7)
    assert predict(x, models, trim=False) == predict(x, loaded, trim=False)
