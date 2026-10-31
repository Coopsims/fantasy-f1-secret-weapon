import numpy as np
import pandas as pd
import pytest

import f1_planner as FP
import season_optimum as SO
from f1_calendar import DEFAULT_CONSTRUCTOR_PRICES, DEFAULT_DRIVER_PRICES


def _official(tmp_path, G=4, seed=0, drop_price=None):
    """Synthetic files in the feed's format: points ~ price, prices drift, names as the feed writes them."""
    rng = np.random.default_rng(seed)
    base = {**DEFAULT_DRIVER_PRICES, **DEFAULT_CONSTRUCTOR_PRICES}
    races = [r for r in SO.RACE_CALENDAR_2026 if not r.get("cancelled")][:G]
    pts, pr = [], []
    for g in range(1, G + 1):
        for name, p0 in base.items():
            typ = "driver" if name in FP.DRIVERS else "constructor"
            price = round(p0 + 0.2 * (g - 1) * rng.choice([-1, 1]), 1)
            if not (drop_price == (g, name)):
                pr.append({"round": races[g - 2]["round"] if g > 1 else 0, "gameday": g, "name": name, "type": typ,
                           "price": price})
            pts.append({"round": races[g - 1]["round"], "gameday": g, "name": name, "type": typ,
                        "points": float(round(p0 * rng.uniform(0.3, 1.8) + rng.normal(0, 8)))})
    (tmp_path / "pts.csv").write_text(pd.DataFrame(pts).to_csv(index=False))
    (tmp_path / "pr.csv").write_text(pd.DataFrame(pr).to_csv(index=False))
    return str(tmp_path / "pts.csv"), str(tmp_path / "pr.csv")


def test_load_official(tmp_path):
    p, c = _official(tmp_path, G=3, drop_price=(2, "Hamilton"))
    gds, names, P, C = SO.load_official(p, c)
    assert gds == [1, 2, 3] and names[0] == "Australian GP"
    assert P.shape == C.shape == (3, len(FP.ENTITIES)) and not np.isnan(C).any()
    h = FP.ENTITIES.index("Hamilton")
    assert C[1, h] == C[0, h]                       # missing price carried forward
    assert SO.load_official(p, c, upto=2)[0] == [1, 2]


def test_optimum_with_limitless_is_exact_and_consistent(tmp_path):
    gds, names, P, C = SO.load_official(*_official(tmp_path, G=4, seed=3))
    best = SO.optimise(P, C, limitless=True, time_limit=120, workers=1, log=lambda *_: None)
    assert all(p.optimal for p, _, _ in best["runs"].values())
    assert best["total"] >= best["base_total"] - 1e-9          # Limitless can only help
    assert best["ceiling"] == pytest.approx(best["total"])        # all proven: nothing can beat it
    assert all(p.bound == pytest.approx(p.objective) for p, _, _ in best["runs"].values())
    ok, rescored = SO.check(best, P, C)
    assert ok and rescored == pytest.approx(best["total"])
    df = SO.season_table(best, gds, names, P, C)
    assert df["points"].sum() == pytest.approx(best["total"], abs=0.5)
    assert len(df) == 4 and (df["bank"] >= -1e-6).all()
    if best["k"] is not None:
        row = df.iloc[best["k"]]
        assert row["chip"] == "limitless"
        assert row["points"] == pytest.approx(FP.limitless_team(P[best["k"]])[2], abs=0.05)
        assert df.iloc[best["k"]]["transfers"] == 0          # the real team is parked that week


def test_limitless_goes_to_the_week_with_the_biggest_unaffordable_haul(tmp_path):
    gds, names, P, C = SO.load_official(*_official(tmp_path, G=3, seed=4))
    P[2, [FP.ENTITIES.index(d) for d in ("Verstappen", "Russell", "Norris", "Piastri", "Antonelli")]] += 200
    best = SO.optimise(P, C, limitless=True, time_limit=120, workers=1, log=lambda *_: None)
    assert best["k"] == 2


def test_an_unfinished_solve_still_gives_a_ceiling(tmp_path):
    gds, names, P, C = SO.load_official(*_official(tmp_path, G=8, seed=6))
    best = SO.optimise(P, C, limitless=False, time_limit=1, workers=1, log=lambda *_: None)
    p = best["plan"]
    assert p.bound is not None and p.bound >= p.total - 1e-6
    assert best["ceiling"] >= best["total"] - 1e-6
