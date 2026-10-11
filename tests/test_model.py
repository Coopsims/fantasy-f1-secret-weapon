import numpy as np
import pytest

import f1_rules as R
from f1_calendar import GRID_2026
from f1_synthetic import synthetic_results
from f1_weekend_model import (
    WeekendForecaster, ModelParams, backtest, fit_model, format_forecast,
)


@pytest.fixture(scope="module")
def season():
    return synthetic_results(seed=5, years=(2024, 2025, 2026), rounds_2026=14)


@pytest.fixture(scope="module")
def forecaster(season):
    return WeekendForecaster(season, verbose=False)


def test_ratings_recover_truth(season, forecaster):
    hist, now_t = forecaster._history(15)
    m = fit_model(hist, dict(GRID_2026), 2026, now_t, ModelParams())
    summ = m.summary().set_index("driver")
    tr = season.truth
    true = {d: tr["team"][(t, 2026)] + tr["skill"][d] for d, t in GRID_2026.items()}
    est = np.array([summ.loc[d, "race_rating"] for d in GRID_2026])
    tru = np.array([true[d] for d in GRID_2026])
    assert np.corrcoef(est, tru)[0, 1] > 0.85
    q = np.array([summ.loc[d, "quali_rating"] for d in GRID_2026])
    assert np.corrcoef(q, tru)[0, 1] > 0.85


def test_calibration_near_truth(forecaster):
    fc = forecaster.forecast(15, n=500)
    c_q = fc.meta["q_cal"][0]
    assert 1.2 < c_q < 2.6            # truth is 1.8
    assert 0.0 <= fc.meta["q_cal"][1] <= 0.85


def test_simulation_invariants(forecaster):
    fc = forecaster.forecast(15, n=3000, seed=1)       # R15: not a sprint weekend
    nD = len(fc.drivers)
    assert fc.sprint is False
    for key in ("quali", "race"):
        srt = np.sort(fc.pos[key], axis=1)
        assert (srt == np.arange(1, nD + 1)).all()      # every sim is a permutation
    assert ((fc.pos["race"] == 1) & ~fc.dnf["race"]).sum(1).max() <= 1
    assert fc.fl["race"].sum(1).min() == 1 and fc.fl["race"].sum(1).max() == 1
    assert (fc.fl["race"] & fc.dnf["race"]).sum() == 0     # DNF can't set fastest lap
    dt = fc.driver_table()
    assert abs(dt["P_win"].sum() - 1.0) < 0.02 or dt["P_win"].sum() > 0.9
    assert abs(dt["P_pole"].sum() - 1.0) < 1e-6
    assert dt["sprint"].abs().sum() == 0                  # no sprint points on a normal weekend
    assert 0.02 < dt["P_dnf"].mean() < 0.2
    assert (fc.dotd.sum(1) == R.DRIVER_OF_THE_DAY).all()  # exactly one DotD per race
    # table totals are the sum of the parts
    assert np.allclose(dt["total"], dt["quali"] + dt["sprint"] + dt["race"])


def test_scoring_recomputed_from_positions(forecaster):
    fc = forecaster.forecast(15, n=200, seed=2)
    q = fc.pos["quali"]
    assert np.allclose(fc.comp[("quali", "finish")] + 0.0,
                       R.score_qualifying(q, None) + np.where(fc.comp[("quali", "finish")] == R.QUALI_NC_PENALTY,
                                                              R.QUALI_NC_PENALTY - R.quali_points(q), 0.0))
    r, dnf = fc.pos["race"], fc.dnf["race"]
    assert np.allclose(fc.comp[("race", "finish")], np.where(dnf, 0, R.race_finish_points(r)))
    assert np.allclose(fc.comp[("race", "dnf")], np.where(dnf, R.DNF_PENALTY_RACE, 0))
    # the winner of the race scored at least the race win points
    assert fc.comp[("race", "finish")].max() == 25


def test_constructor_is_sum_of_driver_parts(forecaster):
    fc = forecaster.forecast(15, n=500, seed=3)
    for ev, comp in (("race", "finish"), ("race", "fastest_lap"), ("race", "overtakes")):
        d = fc.comp[(ev, comp)]
        c = fc.ccomp[(ev, comp)]
        for j, _ in enumerate(fc.teams):
            members = [i for i, ti in enumerate(fc.team_of) if ti == j]
            assert np.allclose(c[:, j], d[:, members].sum(1))
    assert np.isin(fc.ccomp[("quali", "progression")], list(R.CONSTRUCTOR_QUALI_Q2_Q3.values())).all()


def test_sprint_weekend_has_sprint_points(forecaster):
    fc = forecaster.forecast(14, n=1500, seed=4)       # R14 Zandvoort is a sprint weekend
    assert fc.sprint
    dt = fc.driver_table()
    assert dt["sprint"].sum() > 0
    assert (fc.dnf["sprint"].mean() < fc.dnf["race"].mean())      # sprints are safer


def test_conditioning_on_known_qualifying(forecaster):
    order = list(GRID_2026)[::-1]
    fc = forecaster.forecast(15, n=800, seed=5, known={"quali": order})
    dt = fc.driver_table().set_index("driver")
    assert dt.loc[order[0], "P_pole"] == 1.0
    assert dt.loc[order[0], "exp_quali_pos"] == 1.0
    assert dt.loc[order[-1], "exp_quali_pos"] == len(order)
    assert dt.loc[order[0], "quali"] == 10.0


def test_known_pole_boosts_race_forecast(forecaster):
    base = forecaster.forecast(15, n=4000, seed=6)
    worst = base.driver_table().sort_values("total").iloc[0]["driver"]
    rating_order = list(base.driver_table()["driver"])
    cond = [worst] + [d for d in rating_order if d != worst]
    boosted = forecaster.forecast(15, n=4000, seed=6, known={"quali": cond})
    b0 = base.driver_table().set_index("driver").loc[worst]
    b1 = boosted.driver_table().set_index("driver").loc[worst]
    # Starting P1 improves the finishing position and win odds...
    assert b1["exp_race_pos"] < b0["exp_race_pos"]
    assert b1["P_win"] > b0["P_win"]
    # ...but a slow car on pole *loses* fantasy points to the -1/place-lost rule
    assert b1["race"] < b0["race"]


def test_grid_penalty_hurts(forecaster):
    base = forecaster.forecast(15, n=3000, seed=8)
    top = base.driver_table().iloc[0]["driver"]
    pen = forecaster.forecast(15, n=3000, seed=8, grid_penalties={top: 10})
    b = base.driver_table().set_index("driver").loc[top]
    p = pen.driver_table().set_index("driver").loc[top]
    assert p["P_win"] < b["P_win"] and p["exp_race_pos"] > b["exp_race_pos"]
    assert p["P_pole"] == b["P_pole"]            # a grid penalty doesn't change qualifying


def test_excluding_a_driver(forecaster):
    fc = forecaster.forecast(15, n=300, exclude=("Verstappen",))
    assert "Verstappen" not in fc.drivers and len(fc.drivers) == 21


def test_team_samples_and_chips(forecaster):
    fc = forecaster.forecast(15, n=2000, seed=9)
    drivers = ["Verstappen", "Norris", "Leclerc", "Russell", "Albon"]
    cons = ["Ferrari", "McLaren"]
    plain = fc.team_samples(drivers, cons)
    boosted = fc.team_samples(drivers, cons, boost="Norris")
    i = fc.drivers.index("Norris")
    assert np.allclose(boosted - plain, fc.total_points()[:, i])
    nn = fc.team_samples(drivers, cons, no_negative=True)
    assert (nn >= plain - 1e-6).all()
    ch = fc.chip_analysis(drivers, cons, boost="Norris")
    assert ch["no_negative_gain"] >= 0 and set(ch["boost"]) == set(drivers)


def test_format_and_json(forecaster):
    fc = forecaster.forecast(14, n=300)
    txt = format_forecast(fc, top_n=5)
    assert "SPRINT" in txt and "CONSTRUCTORS" in txt
    d = fc.to_dict()
    assert len(d["drivers"]) == 22 and len(d["constructors"]) == 11
    assert set(fc.breakdown("Norris")).issubset({"quali", "sprint", "race"})
    assert "pit" in fc.breakdown("McLaren")["race"]


def test_no_history_falls_back_to_price_prior():
    from f1_weekend_data import Results
    import pandas as pd
    empty = Results(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    f = WeekendForecaster(empty, verbose=False)
    fc = f.forecast(1, n=2000, seed=1)
    ct = fc.constructor_table().set_index("team")
    # pre-season prices rank the big three above the back markers
    assert ct.loc["Mercedes", "total"] > ct.loc["Cadillac", "total"]
    assert ct.loc["McLaren", "total"] > ct.loc["Audi", "total"]


def test_backtest_beats_naive_and_is_calibrated():
    res = synthetic_results(seed=21, years=(2024, 2025, 2026), rounds_2026=24)
    df = backtest(res, years=(2026,), n_sims=1200, verbose=False, max_weekends=10)
    assert len(df) == 10
    assert df["mae_sim"].mean() < df["mae_naive"].mean()
    assert df["corr_sim"].mean() > df["corr_naive"].mean()
    assert 0.65 < df["coverage80"].mean() < 0.97
    assert df["quali_rank_corr"].mean() > 0.7
