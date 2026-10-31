import shutil

import numpy as np
import pytest

import f1_planner as FP
import season_model_test as SMT
import season_optimum as SO
from tests.test_season_optimum import _official


def _noisy(P, H=3, seed=0, sd=10.0):
    rng = np.random.default_rng(seed)
    return [P[t:t + H] + rng.normal(0, sd, P[t:t + H].shape) for t in range(len(P))]


def _rounds(G):
    return [[r["round"] for r in SO.RACE_CALENDAR_2026 if not r.get("cancelled")][t:t + 3] for t in range(G)]


def test_without_limitless_it_is_the_earlier_replay(tmp_path, monkeypatch):
    from experiments.s26_season_replay import play as old_play
    p, c = _official(tmp_path, G=5, seed=1)
    shutil.copy(p, tmp_path / "fantasy_f1_official_points.csv")
    shutil.copy(c, tmp_path / "fantasy_f1_prices_official.csv")
    monkeypatch.chdir(tmp_path)
    gds, names, P, C = SO.load_official()
    fc = _noisy(P)
    old, old_log = old_play({t + 1: (None, fc[t]) for t in range(len(P))})
    new, log = SMT.play(fc, _rounds(len(P)), P, C)
    assert new == pytest.approx(old)
    assert list(log["chip"]) == [c if isinstance(c, str) else "" for c in old_log["chip"]]


def test_perfect_forecasts_over_the_whole_season_reach_the_optimum(tmp_path):
    """With the real points as forecasts, the full season as the window and no discount, playing week by week is the
    hindsight optimum, Limitless included (the plan stays optimal as it rolls)."""
    gds, names, P, C = SO.load_official(*_official(tmp_path, G=4, seed=3))
    best = SO.optimise(P, C, limitless=True, time_limit=120, workers=1, log=lambda *_: None)
    fc = [P[t:] for t in range(len(P))]
    rounds = [[gds[i] for i in range(t, len(P))] for t in range(len(P))]
    total, log = SMT.play(fc, rounds, P, C, limitless="window", discount=1.0, gap=0.0, time_limit=120)
    assert total == pytest.approx(best["total"])
    assert ("limitless" in set(log["chip"])) == (best["k"] is not None)
    assert log["points"].sum() == pytest.approx(total)


def test_sprint_rule_and_autopilot(tmp_path):
    gds, names, P, C = SO.load_official(*_official(tmp_path, G=6, seed=5))
    rounds = _rounds(len(P))
    total, log = SMT.play(_noisy(P, seed=2), rounds, P, C, limitless="sprint", autopilot=True)
    lim = log.index[log["chip"] == "limitless"]
    assert len(lim) <= 1 and all(SMT.SPRINT[rounds[t][0]] for t in lim)
    ap = log.index[log["chip"] == "autopilot"]
    assert len(ap) <= 1
    for t in ap:                                     # the 2x went to the team's top-scoring driver
        drivers = [FP.ENTITIES.index(d) for d in log.loc[t, "drivers"].split(", ")]
        assert log.loc[t, "boost"] == FP.ENTITIES[max(drivers, key=lambda e: P[t][e])]
    assert set(log["chip"]) - {""} <= {"wildcard", "3x_booster", "no_negative", "limitless", "autopilot"}
    assert (log["chip"].replace("", np.nan).dropna().value_counts() == 1).all()     # each chip once
