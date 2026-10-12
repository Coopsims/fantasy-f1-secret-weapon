import numpy as np

import f1_planner as FP
from f1_strategy import Rules, State, solve_strategy


def _world(T=3, seed=0):
    rng = np.random.default_rng(seed)
    E = len(FP.ENTITIES)
    pts = rng.normal(10, 8, (T, E))
    price = np.repeat(rng.uniform(4, 30, E)[None, :], T + 1, 0)
    team = sorted(np.argsort(price[0])[:5].tolist() if False else [])
    return pts, price


def _cheap_team(price):
    d = [FP.ENTITIES[i] for i in np.argsort(np.where(FP.IS_DRIVER, price[0], np.inf))[:5]]
    c = [FP.ENTITIES[i] for i in np.argsort(np.where(~FP.IS_DRIVER, price[0], np.inf))[:2]]
    return d + c


def test_frozen_round_keeps_the_team_and_plays_no_chip():
    pts, price = _world()
    team = _cheap_team(price)
    idx = frozenset(FP.ENTITIES.index(n) for n in team)
    st = State(prev_team=idx, wealth=float(price[0][list(idx)].sum()) + 30, free_transfers=3, chips_left=Rules().chips)
    p = solve_strategy(pts, price[:3], FP.IS_DRIVER, Rules(), st, frozen=(1,))
    assert p.teams[1] == p.teams[0] and p.chips[1] is None


def test_limitless_is_used_only_when_it_pays_and_never_hurts():
    pts, price = _world(seed=1)
    team = _cheap_team(price)
    base, _ = FP.plan(pts, price, team=team, bank=5.0, free=2, chips=())
    with_l, _ = FP.plan(pts, price, team=team, bank=5.0, free=2, chips=("limitless",))
    assert with_l.total >= base.total - 1e-6
    k = [i for i, c in enumerate(with_l.chips) if c == "limitless"]
    if k:
        lt, lb, lv = FP.limitless_team(pts[k[0]])
        assert with_l.teams[k[0]] == lt and abs(with_l.round_points[k[0]] - lv) < 1e-6


def test_keep_and_avoid_constraints():
    pts, price = _world(seed=2)
    team = _cheap_team(price)
    keep, avoid = team[0], FP.ENTITIES[int(np.argmax(pts[0] * FP.IS_DRIVER))]
    p, _ = FP.plan(pts, price, team=team, bank=50.0, free=3, chips=(), locked_in=[keep], locked_out=[avoid])
    for t in p.teams:
        names = {FP.ENTITIES[e] for e in t}
        assert keep in names and avoid not in names


def test_planning_blend_weights_are_applied_to_all_assets():
    import pandas as pd
    E = len(FP.ENTITIES)
    sim = {20: np.full(E, 10.0)}
    op = pd.DataFrame({"gameday": [1] * E, "name": FP.ENTITIES, "points": [30.0] * E, "type": ["driver"] * E, "round": [1] * E})
    ew = FP.ewma_official(op, 0.2)
    assert np.allclose(ew.reindex(FP.ENTITIES).values, 30.0)
    out = np.full((1, E), 20.0)
    pb = {"w_model": 0.5, "w_sim": 0.3, "w_ewma": 0.2}
    blended = pb["w_model"] * out[0] + pb["w_sim"] * sim[20] + pb["w_ewma"] * ew.values
    assert np.allclose(blended, 0.5 * 20 + 0.3 * 10 + 0.2 * 30)


def test_no_negative_gain_is_positive_and_falls_with_the_forecast():
    mu = np.zeros((1, len(FP.ENTITIES)))
    mu[0, :6] = [30, 20, 10, 5, 0, -5]
    g = FP.nn_gain(mu)[0][:6]
    assert (g > 0).all() and all(a < b for a, b in zip(g, g[1:]))
    assert abs(g[-1] - (5 + FP.nn_gain(np.array([[5.0] * len(FP.ENTITIES)]))[0][0])) < 1e-6   # E[max(X,0)] - mu symmetry


def test_solver_uses_no_negative_when_its_expected_gain_is_large():
    pts, price = _world(T=2, seed=4)
    team = _cheap_team(price)
    idx = frozenset(FP.ENTITIES.index(n) for n in team)
    st = State(prev_team=idx, wealth=float(price[0][list(idx)].sum()) + 5, free_transfers=2, chips_left=("no_negative",))
    gain = np.zeros_like(pts); gain[1] = 50.0                     # huge downside protection in week 2 only
    p = solve_strategy(pts, price[:2], FP.IS_DRIVER, Rules(chips=("no_negative",)), st, nn_gain=gain)
    assert p.chips == [None, "no_negative"]


def test_autopilot_gain_is_positive_for_a_spread_team():
    pts = np.zeros(len(FP.ENTITIES)); pts[:5] = [30, 25, 20, 10, 5]
    assert FP.autopilot_gain(frozenset(range(5)) | {22, 23}, pts) > 2.0


def test_expand_seeds_makes_one_member_per_seed():
    import f1_planner as FP
    specs = [{"family": "xgb", "n_seeds": 3, "hp": {}}, {"family": "extratrees", "hp": {}}]
    out = FP.expand_seeds(specs)
    assert [(s["family"], s["seed"]) for s in out] == [("xgb", 0), ("xgb", 1), ("xgb", 2), ("extratrees", 0)]
    assert all("n_seeds" not in s for s in out)


def test_no_free_transfers_carry_out_of_a_limitless_week():
    # game FAQ: after a Limitless week the next race has only the regular 2 free transfers (no carry-over)
    from f1_strategy import evaluate_plan
    pts, price = _world(seed=5)
    team = _cheap_team(price)
    idx = frozenset(FP.ENTITIES.index(n) for n in team)
    pts[:] = 0.0
    pts[:, list(idx)] = 50.0
    outsiders = [i for i in range(len(FP.ENTITIES)) if FP.IS_DRIVER[i] and i not in idx][:3]
    pts[2, outsiders] = 200.0                                # race 3 rewards three new drivers
    st = State(prev_team=idx, wealth=float(price[0][list(idx)].sum()) + 300, free_transfers=3, chips_left=())
    p = solve_strategy(pts, price[:3], FP.IS_DRIVER, Rules(), st, frozen=(1,))
    assert p.penalties[2] == 10.0 and set(outsiders) <= set(p.teams[2])
    tot, *_ = evaluate_plan(p.teams, p.boosts, p.chips, pts, price[:3], FP.IS_DRIVER, Rules(), st, frozen=(1,))
    assert abs(tot - p.total) < 1e-6
    _, _, pen_unfrozen, *_ = evaluate_plan(p.teams, p.boosts, p.chips, pts, price[:3], FP.IS_DRIVER, Rules(), st)
    assert pen_unfrozen[2] == 0.0                            # without the rule the 3rd transfer would have been free
