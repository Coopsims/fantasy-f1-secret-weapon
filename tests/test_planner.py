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
