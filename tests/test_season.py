import numpy as np
import pytest

import f1_season as S
from f1_strategy import Rules, evaluate_plan
from f1_synthetic import synthetic_results


@pytest.fixture(scope="module")
def run():
    res = synthetic_results(seed=9, rounds_2026=5)
    rep = S.replay(res, 2026, n_sims=800, horizon=2, verbose=False)
    prices, _ = S.price_table(rep["actuals"]["rounds"], rep["actuals"]["pts"])
    return res, rep, prices


def test_replay_uses_only_earlier_weekends(run):
    res, rep, _ = run
    acc = rep["accuracy"]
    assert list(acc["round"]) == [1, 2, 3, 4, 5]
    # each step is trained on exactly the weekends before it - one more every round
    assert list(np.diff(acc["history_weekends"])) == [1, 1, 1, 1]
    first_2026 = res.weekends[res.weekends["year"] == 2026]["t"].min()
    assert acc["history_weekends"].iloc[0] == first_2026
    t_last = rep["actuals"]["t"][-1]
    assert len(S.truncate(res, t_last).weekends) == t_last


def test_forecasts_do_not_change_when_the_future_is_deleted(run):
    """Strongest leakage test: forecasts for round 3 are identical if rounds 3+ are removed."""
    res, rep, _ = run
    t3 = rep["actuals"]["t"][2]
    cut = S.truncate(res, t3)
    from f1_weekend_model import WeekendForecaster, ModelParams
    from dataclasses import replace
    f = WeekendForecaster(cut, params=replace(ModelParams(), dotd=False), year=2026, verbose=False)
    fc = f.forecast(3, n=800, seed=103)
    expect = rep["forecasts"][(3, 3)]["mean"][: len(S.DRIVERS)]
    got = np.array([fc.total_points(include_dotd=False).mean(0)[fc.drivers.index(d)] for d in S.DRIVERS])
    assert np.allclose(got, expect)


def test_hindsight_optimum_beats_model_and_is_consistent(run):
    _, rep, prices = run
    rules = Rules()
    play = S.play_season(rep, prices, rules, horizon=2, verbose=False)
    plan, opt = S.hindsight_optimum(rep["actuals"], prices, rules, time_limit=120)
    assert plan.optimal
    assert plan.total >= play["total"] - 1e-6
    tot, *_ , ok = evaluate_plan(plan.teams, plan.boosts, plan.chips, rep["actuals"]["pts"], prices,
                                 S.IS_DRIVER, rules)
    assert ok and tot == pytest.approx(plan.total)
    assert opt["points"].sum() == pytest.approx(plan.total)


def test_model_play_is_feasible_every_week(run):
    _, rep, prices = run
    play = S.play_season(rep, prices, Rules(), horizon=2, verbose=False)
    assert len(play["log"]) == 5
    assert all(len(t) == 7 for t in play["log"]["team"])
    assert play["log"]["chip"].dropna().is_unique


def test_projected_prices_follow_the_rule():
    from f1_prices import PriceRule, simulate
    rng = np.random.default_rng(0)
    start = rng.uniform(5, 30, 10).round(1)
    pts = rng.normal(15, 10, (5, 10))
    rule = PriceRule()
    P = simulate(pts, start, rule)
    past = [pts[j] / P[j] for j in range(2)]           # races 1-2 already run
    proj = S.project_prices(P[2], past, pts[2:5], rule)
    assert np.allclose(proj, P[2:6])


def test_policy_variants_play_and_tune(run):
    _, rep, prices = run
    from f1_prices import PriceRule
    grid = [S.PlayPolicy(horizon=1, price_aware=False), S.PlayPolicy(horizon=2, value_weight=5.0),
            S.PlayPolicy(horizon=2, chip_cost=20.0, ewma_blend=0.3)]
    df = S.tune_play(rep, prices, Rules(), PriceRule(), grid, workers=1, verbose=False)
    assert len(df) == 3 and (df["delta"] >= -1e-6).all()
    assert np.allclose(df["delta_first_half"] + df["delta_second_half"], df["delta"])


def test_price_csv_names_are_canonicalised(tmp_path):
    import numpy as np
    import pandas as pd
    import f1_season as S
    rounds = [1, 2]
    pts = np.zeros((2, len(S.ENTITIES)))
    rows = [(0, n, 10.0) for n in S.ENTITIES if n != "Haas F1 Team"] + [(0, "Haas", 9.9)]
    p = tmp_path / "prices.csv"
    pd.DataFrame(rows, columns=["round", "name", "price"]).to_csv(p, index=False)
    prices, _ = S.price_table(rounds, pts, str(p))
    assert prices[0][S.ENTITIES.index("Haas F1 Team")] == 9.9      # "Haas" in the CSV is the Haas F1 Team constructor


def test_replay_cache_is_keyed_on_the_data():
    import f1_season as S
    from f1_synthetic import synthetic_results
    a = synthetic_results(seed=1, rounds_2026=4)
    b = synthetic_results(seed=2, rounds_2026=4)
    assert len(a.weekends) == len(b.weekends) and S._data_key(a) != S._data_key(b)


def test_replay_cache_key_changes_with_official_points():
    import pandas as pd
    import f1_season as S
    from f1_synthetic import synthetic_results
    r = synthetic_results(seed=1, rounds_2026=4)
    off = pd.DataFrame({"round": [1], "name": ["Verstappen"], "points": [30.0]})
    assert S._replay_key(r, 2026, 100, 3, None, None) != S._replay_key(r, 2026, 100, 3, None, off)


def test_play_includes_pit_points_when_scored_on_official_points(run, monkeypatch):
    """With official points constructors score pit stops, so the planner must see the pit forecast too."""
    _, rep, prices = run
    seen = []
    real = S.solve_strategy

    def spy(pts, *a, **k):
        seen.append(np.array(pts))
        return real(pts, *a, **k)

    monkeypatch.setattr(S, "solve_strategy", spy)
    first = rep["forecasts"][(rep["actuals"]["rounds"][0], rep["actuals"]["rounds"][0])]
    for official in (False, True):
        seen.clear()
        S.play_season({**rep, "official": official}, prices, Rules(), horizon=1, verbose=False)
        expect = first["mean"] + (first["pit"] if official else 0.0)
        assert np.allclose(seen[0][0], expect)
    assert first["pit"][~S.IS_DRIVER].sum() > 0           # the two runs really differ


def test_tune_play_reports_grid_position(run):
    _, rep, prices = run
    from f1_prices import PriceRule
    grid = [S.PlayPolicy(horizon=2), S.PlayPolicy(horizon=1)]
    df = S.tune_play(rep, prices, Rules(), PriceRule(), grid, workers=1, verbose=False)
    assert sorted(df["grid_index"]) == [0, 1]
    assert list(df.set_index("grid_index").loc[[0, 1], "horizon"]) == [2, 1]
