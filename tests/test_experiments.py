"""The experiment harness must not look at the future, and the opt-in model options must be inert by default."""
import numpy as np
import pytest

import mlflow_experiments as X
from f1_synthetic import synthetic_results
from f1_weekend_model import ModelParams, backtest


@pytest.fixture(scope="module")
def world():
    res = synthetic_results(seed=5, years=(2024, 2025), rounds_2026=None)
    truth, ctruth = X.build_truth(res)
    return res, truth, ctruth


def _targets(res, n=3):
    wk = res.weekends
    return wk[[(wk["t"] < t).sum() >= 8 for t in wk["t"]]].tail(n).reset_index(drop=True)


@pytest.mark.parametrize("make", [lambda: X.Baseline("last3"), lambda: X.Baseline("ewma", 0.3),
                                  lambda: X.Baseline("season"), lambda: X.GBM(n_estimators=20, min_train_rows=50)])
def test_forecasts_ignore_the_target_weekend_and_everything_after(world, make):
    res, truth, ctruth = world
    tg = _targets(res)
    t0 = int(tg["t"].iloc[0])
    clean = make().predict(res, truth, ctruth, tg.iloc[[0]]).drivers
    tampered = truth.copy()
    future = tampered["t"] >= t0                       # scramble the target weekend and all later ones
    for c in ("total", "quali", "race", "sprint"):
        tampered.loc[future, c] = tampered.loc[future, c].values[::-1] * 3 + 11
    tampered.loc[future, "race_dnf"] = ~tampered.loc[future, "race_dnf"]
    again = make().predict(res, tampered, ctruth, tg.iloc[[0]]).drivers
    m = clean.merge(again, on=["t", "driver"], suffixes=("_a", "_b"))
    assert len(m) > 10
    assert np.allclose(m["pred_a"], m["pred_b"])


def test_rolling_nnls_uses_only_earlier_weekends_and_weights_sum_to_one(world):
    import pandas as pd
    from experiments.ens import rolling_nnls
    res, truth, _ = world
    rng = np.random.default_rng(0)
    w = truth[["t", "driver", "team", "total"]].copy()
    w["a"] = w["total"] + rng.normal(0, 3, len(w))
    w["b"] = w["total"] * 0.5 + rng.normal(0, 3, len(w))
    ts = sorted(w["t"].unique())[10:14]
    fc, wts = rolling_nnls(w, ["a", "b"], ts, min_rows=50)
    assert np.allclose(wts[["a", "b"]].sum(axis=1), 1.0) and (wts[["a", "b"]] >= 0).all().all()
    w2 = w.copy()
    w2.loc[w2["t"] >= ts[2], ["a", "b", "total"]] *= 7        # change the future relative to ts[2]
    fc2, wts2 = rolling_nnls(w2, ["a", "b"], ts, min_rows=50)
    assert np.allclose(wts[wts["t"] <= ts[2]]["a"], wts2[wts2["t"] <= ts[2]]["a"])


def test_default_params_give_the_same_backtest_with_or_without_new_options(world):
    res, *_ = world
    a = backtest(res, years=(2025,), n_sims=300, verbose=False, max_weekends=4)
    b = backtest(res, years=(2025,), n_sims=300, params=ModelParams(team_block=0, p0_max=0.15), verbose=False, max_weekends=4)
    assert np.allclose(a["mae_sim"], b["mae_sim"])


def test_team_block_option_runs_and_changes_the_fit(world):
    res, *_ = world
    a = backtest(res, years=(2025,), n_sims=300, verbose=False, max_weekends=3)
    b = backtest(res, years=(2025,), n_sims=300, params=ModelParams(team_block=4), verbose=False, max_weekends=3)
    assert len(b) == len(a) and np.isfinite(b["mae_sim"]).all()
