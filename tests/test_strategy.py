import numpy as np
import pytest

from f1_strategy import Rules, State, solve_strategy, brute_force_strategy, evaluate_plan


def _instance(seed, T=3, nd=4, nc=3, tight=0.8):
    rng = np.random.default_rng(seed)
    E = nd + nc
    isd = np.array([True] * nd + [False] * nc)
    pts = rng.normal(6, 14, (T, E))
    base = np.r_[rng.uniform(8, 20, nd), rng.uniform(8, 20, nc)]
    price = np.round(base[None, :] + np.cumsum(rng.normal(0, 0.8, (T, E)), axis=0), 1)
    rules = Rules(n_drivers=2, n_constructors=1, budget=float(np.sort(base)[-3:].sum() * tight),
                  penalty=10, free_per_round=1, max_bank=2)
    return pts, price, isd, rules


@pytest.mark.parametrize("seed", range(8))
def test_milp_matches_exhaustive_enumeration(seed):
    pts, price, isd, rules = _instance(seed, tight=0.75 + 0.03 * seed)
    bf_total, _ = brute_force_strategy(pts, price, isd, rules)
    plan = solve_strategy(pts, price, isd, rules)
    assert plan.optimal
    assert plan.total == pytest.approx(bf_total, abs=1e-6)
    assert plan.objective == pytest.approx(bf_total, abs=1e-6)


def test_matches_enumeration_from_a_mid_season_state():
    pts, price, isd, rules = _instance(3)
    state = State(prev_team=frozenset({0, 1, 4}), wealth=float(price[0, [0, 1, 4]].sum() + 1.5),
                  free_transfers=1, chips_left=("3x_booster", "wildcard"))
    bf_total, _ = brute_force_strategy(pts, price, isd, rules, state)
    plan = solve_strategy(pts, price, isd, rules, state)
    assert plan.total == pytest.approx(bf_total, abs=1e-6)
    assert "no_negative" not in plan.chips


def test_plan_is_feasible_and_self_consistent():
    pts, price, isd, rules = _instance(5, T=4)
    plan = solve_strategy(pts, price, isd, rules)
    total, per, pens, trs, ok = evaluate_plan(plan.teams, plan.boosts, plan.chips, pts, price, isd, rules, plan_boosts3=plan.boosts3)
    assert ok and total == pytest.approx(plan.total)
    assert sum(1 for c in plan.chips if c) == len([c for c in plan.chips if c])      # chips unique
    assert len({c for c in plan.chips if c}) == len([c for c in plan.chips if c])
    assert all(b in t for b, t in zip(plan.boosts, plan.teams))


def test_never_worse_than_any_fixed_team():
    pts, price, isd, rules = _instance(2, T=4)
    plan = solve_strategy(pts, price, isd, rules)
    # holding one fixed affordable team all season (boosting its best driver each week)
    best_fixed = -np.inf
    for team in [frozenset({a, b, c}) for a in range(4) for b in range(a + 1, 4) for c in (4, 5, 6)]:
        boosts = [max((e for e in team if isd[e]), key=lambda e: pts[t, e]) for t in range(4)]
        tot, *_ , ok = evaluate_plan([team] * 4, boosts, [None] * 4, pts, price, isd, rules)
        if ok:
            best_fixed = max(best_fixed, tot)
    assert plan.total >= best_fixed - 1e-9


def test_transfer_penalty_is_charged():
    # one entity is hugely better in week 2 only; with zero free transfers a swap costs 10
    T, nd, nc = 2, 3, 2
    isd = np.array([True] * nd + [False] * nc)
    pts = np.zeros((T, nd + nc))
    pts[0, 0], pts[1, 1] = 5, 12            # driver0 good wk1, driver1 good wk2
    price = np.full((T, nd + nc), 10.0)
    rules = Rules(n_drivers=1, n_constructors=1, budget=20.0, penalty=10, free_per_round=0, max_bank=0, chips=())
    # start already holding driver0 + constructor 3; with prev_team=None the first pick is free and
    # "driver1 from week 1" ties the swap plan at 24 without any penalty
    start = State(prev_team=frozenset({0, nd}), free_transfers=0, chips_left=())
    plan = solve_strategy(pts, price, isd, rules, start)
    # boost doubles: stay on driver0 = 10 + 0 ; swap in wk2 = 10 + 24 - 10 = 24 ; swap in wk1 = 0 + 24 - 10 = 14
    assert plan.total == pytest.approx(5 * 2 + 12 * 2 - 10)
    assert plan.penalties[1] == 10 and plan.transfers_in[1] == 1


def test_value_weight_prefers_rising_assets_on_ties():
    isd = np.array([True, True, False])
    pts = np.array([[10.0, 10.0, 5.0]])
    price = np.array([[10.0, 10.0, 10.0]])
    terminal = np.array([9.4, 10.6, 10.0])           # driver 1 is about to rise
    rules = Rules(n_drivers=1, n_constructors=1, budget=30.0, chips=())
    st = State(free_transfers=1, chips_left=())
    plain = solve_strategy(pts, price, isd, rules, st)
    valued = solve_strategy(pts, price, isd, rules, st, terminal_price=terminal, value_weight=10.0)
    assert 1 in valued.teams[0] and 0 not in valued.teams[0]
    assert valued.total == plain.total                 # true points unchanged by the planning aid


def test_chip_cost_keeps_chips_in_reserve():
    pts, price, isd, rules = _instance(1, T=3)
    free = solve_strategy(pts, price, isd, rules)
    held = solve_strategy(pts, price, isd, rules, chip_cost={c: 1e4 for c in rules.chips})
    assert any(free.chips) and not any(held.chips)
