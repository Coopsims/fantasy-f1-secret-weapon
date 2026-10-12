import numpy as np
import pandas as pd
import pytest

from f1_prices import (PriceRule, step_prices, simulate, checkpoints_from_csv, fit_rule,
                       reconstruct, holdout_check)


def test_guide_worked_example():
    # 2025 guide: $23.0m driver scoring 10 in Australia -> PPM 0.43 -> terrible -> -$0.3m (A tier)
    assert step_prices(np.array([23.0]), [np.array([10 / 23.0])], PriceRule())[0] == pytest.approx(22.7)


def test_tiers_levels_and_floor():
    r = PriceRule()
    ppm = np.array([[0.5, 0.7, 1.0, 1.5, 0.5, 0.7, 1.0, 1.5, 0.0]])
    price = np.array([25, 25, 25, 25, 10, 10, 10, 10, 4.1])
    out = step_prices(price, ppm, r)
    assert list(out) == [24.7, 24.9, 25.1, 25.3, 9.4, 9.8, 10.2, 10.6, 4.0]


def test_early_season_equivalence_with_zero_padding():
    # padded rule: 3-slot average with zeros vs round-n cut-offs n*(0.2, 0.3, 0.4)
    rng = np.random.default_rng(0)
    for n in (1, 2):
        ppm = rng.uniform(0, 2, (n, 50))
        padded = np.vstack([ppm, np.zeros((3 - n, 50))]).mean(0)
        lvl_pad = np.searchsorted(np.array([0.2, 0.3, 0.4]) * n, padded, side="right")
        lvl_avg = np.searchsorted(np.array([0.6, 0.9, 1.2]), ppm.mean(0), side="right")
        assert (lvl_pad == lvl_avg).all()


def test_window_uses_price_paid_for_each_race():
    pts = np.array([[30.0], [0.0], [0.0], [0.0]])
    P = simulate(pts, np.array([10.0]), PriceRule())
    # race 1: 3.0 ppm -> great (+0.6); race 2 avg (3.0+0)/2 -> great; race 3 avg 1.0 -> good;
    # race 4 drops race 1 from the window -> terrible
    assert list(P[:, 0]) == [10.0, 10.6, 11.2, 11.4, 10.8]


def test_cancelled_rounds_map_to_raced_weekends():
    names = ["A", "B"]
    df = pd.DataFrame({"round": [0, 0, 3, 3, 6, 6], "name": names * 3, "price": [10, 20, 11, 19, 12, 18]})
    cps = checkpoints_from_csv(df, [1, 2, 3, 6, 7], names)       # rounds 4, 5 cancelled
    assert sorted(cps) == [0, 3, 4]
    assert list(cps[4]) == [12, 18]


def test_fit_recovers_rule_from_two_real_weekends():
    rng = np.random.default_rng(3)
    start = rng.uniform(5, 30, 33).round(1)
    pts = np.maximum(-20, start * rng.uniform(0.5, 1.6, 33) + rng.normal(0, 12, (12, 33)))
    true = PriceRule(tier_cut=19.0, cuts_a=(0.66, 0.99, 1.32), cuts_b=(0.66, 0.99, 1.32))
    P = simulate(pts, start, true)
    Q, info = reconstruct(pts, start, {0: P[0], 1: P[1], 2: P[2]}, verbose=False)
    assert info["fitted"] and info["fit_mae"] == pytest.approx(0, abs=1e-9)
    assert np.abs(Q - P).max() < 0.15


def test_partial_checkpoints_and_holdout():
    rng = np.random.default_rng(4)
    start = rng.uniform(5, 30, 20).round(1)
    pts = np.maximum(-20, start * rng.uniform(0.5, 1.6, 20) + rng.normal(0, 12, (10, 20)))
    P = simulate(pts, start, PriceRule())
    part = P[5].copy()
    part[::2] = np.nan                                   # only some assets known
    anchors = {0: P[0], 2: P[2], 5: part, 10: P[10]}
    Q, _ = reconstruct(pts, start, anchors, verbose=False)
    assert np.allclose(Q[5][1::2], P[5][1::2]) and np.allclose(Q[10], P[10])
    ho = holdout_check(pts, anchors)
    assert len(ho) == 3 and (ho["mae"] < 0.1).all()


def test_no_checkpoints_uses_defaults():
    pts = np.full((3, 2), 10.0)
    Q, info = reconstruct(pts, np.array([10.0, 25.0]), {}, verbose=False)
    assert not info["fitted"] and Q.shape == (4, 2)
