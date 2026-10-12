#!/usr/bin/env python3
"""
Exact season strategy optimiser (f1_strategy.py)
================================================
Given the points every driver / constructor scored each weekend and the price
they could be bought at, find the **provably optimal** sequence of teams,
boosts, transfers and chips.

Why a MILP rather than literal enumeration?  There are ~1.4 million legal
5-driver + 2-constructor teams per weekend, the budget at each weekend depends on
the price gains of the team held before it, and transfers / free-transfer banking
couple the weekends - the search space is astronomically large.  A mixed-integer
program solved by branch-and-bound (HiGHS via scipy) searches it exhaustively
*with a proof of optimality*.  `brute_force_strategy` (true enumeration) is kept
for tiny instances and the tests assert both agree.

Rules modelled (all configurable in `Rules`)
  * N_DRIVERS drivers + N_CONSTRUCTORS constructors, budget = $100m at the start;
    thereafter wealth moves with the price changes of the team held
    (buy/sell at the current price, no spread).
  * one driver is boosted x2 every weekend.
  * transfers: `free_per_round` free transfers a weekend, bank capped at
    `max_bank`; each transfer beyond the free ones costs `penalty` points.
  * chips (each once, at most one a weekend): wildcard (unlimited free
    transfers), 3x booster (boost is x3), no negative (negative scores -> 0).
  * Final Fix and Autopilot are NOT modelled (they depend on in-weekend information).
"""

from dataclasses import dataclass, field
from itertools import combinations, product

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix


@dataclass
class Rules:
    n_drivers: int = 5
    n_constructors: int = 2
    budget: float = 100.0
    penalty: float = 10.0
    free_per_round: int = 2
    max_bank: int = 3
    chips: tuple = ("wildcard", "3x_booster", "no_negative")


@dataclass
class State:
    """Where the team stands before the first weekend being planned."""
    prev_team: frozenset = None      # None => free initial pick
    wealth: float = None             # team value at round-0 prices + bank; None => rules.budget
    free_transfers: int = 2
    chips_left: tuple = ("wildcard", "3x_booster", "no_negative")


@dataclass
class Plan:
    teams: list                       # frozenset of entity indices per round
    boosts: list                      # boosted driver index per round
    chips: list                       # None | "wildcard" | "3x_booster" | "no_negative"
    transfers_in: list
    penalties: list
    round_points: list
    total: float
    objective: float = None
    status: str = ""
    optimal: bool = False


def _chip_pts(vals, chip):
    return np.maximum(vals, 0.0) if chip == "no_negative" else vals


def evaluate_plan(plan_teams, plan_boosts, plan_chips, pts, price, is_driver, rules, state=None,
                  discount=None):
    """
    Score a plan under the rules.  Returns (total, per_round_points, penalties, transfers,
    feasible).  This is the single definition of the rules that the MILP and the brute
    force are both checked against.
    """
    state = state or State(free_transfers=rules.free_per_round, chips_left=rules.chips)
    T = len(plan_teams)
    W = state.wealth if state.wealth is not None else rules.budget
    ft = state.free_transfers
    prev = state.prev_team
    used_chip = set()
    total, per_round, pens, trs = 0.0, [], [], []
    for t in range(T):
        team, boost, chip = plan_teams[t], plan_boosts[t], plan_chips[t]
        if chip is not None:
            if chip in used_chip or chip not in state.chips_left:
                return -np.inf, [], [], [], False
            used_chip.add(chip)
        cost = sum(price[t][e] for e in team)
        if cost > W + 1e-9:
            return -np.inf, [], [], [], False
        n_in = 0 if prev is None else len(team - prev)
        pen = 0.0
        if prev is not None and chip != "wildcard":
            pen = rules.penalty * max(0, n_in - ft)
        # free-transfer bank for the next weekend
        used = 0 if (prev is None or chip == "wildcard") else min(n_in, ft)
        ft = min(rules.max_bank, ft - used + rules.free_per_round)
        v = np.array([pts[t][e] for e in sorted(team)], dtype=float)
        v = _chip_pts(v, chip)
        bp = _chip_pts(np.array([pts[t][boost]]), chip)[0]
        mult = 2.0 if chip == "3x_booster" else 1.0
        r = float(v.sum() + mult * bp - pen)
        w = 1.0 if discount is None else discount[t]
        total += w * r
        per_round.append(r)
        pens.append(pen)
        trs.append(n_in)
        # wealth after the weekend's price moves
        if t + 1 < T:
            W = W + sum(price[t + 1][e] - price[t][e] for e in team)
        prev = team
    return total, per_round, pens, trs, True


def solve_strategy(pts, price, is_driver, rules=None, state=None, discount=None,
                   time_limit=300, mip_rel_gap=0.0, verbose=False,
                   terminal_price=None, value_weight=0.0, chip_cost=None, frozen=(), nn_gain=None):
    """
    Maximise (discounted) points over all weekends.

    pts, price : arrays (T, E) - points scored / price paid before each weekend.
    is_driver  : bool array (E,)
    terminal_price, value_weight : planning aids for a rolling horizon - reward each $1m of
        team-value growth over the horizon (price[t+1] - price[t] of every team held, with
        terminal_price after the last weekend) with `value_weight` points.
    chip_cost  : {chip: points} subtracted when a chip is used (keeps chips in reserve
        unless the planned gain beats it).
    frozen     : rounds in which the team cannot change and no chip can be played (used to model the
        Limitless chip: the real team is parked that week and reverts afterwards; see f1_planner).
    nn_gain    : optional (T, E) expected gain of the No Negative chip per held asset when `pts` are forecasts -
        E[max(X, 0)] - E[X] under the forecast error (the default, max(-pts, 0), is right only for known points).
    Neither changes how a plan is *scored*: plan.total is always true points (evaluate_plan).
    Returns a Plan (plan.optimal is True when optimality is proven).
    """
    pts = np.asarray(pts, float)
    price = np.asarray(price, float)
    is_driver = np.asarray(is_driver, bool)
    rules = rules or Rules()
    state = state or State(free_transfers=rules.free_per_round, chips_left=rules.chips)
    T, E = pts.shape
    D = np.flatnonzero(is_driver)
    disc = np.ones(T) if discount is None else np.asarray(discount, float)
    chips_avail = set(state.chips_left) & set(rules.chips)
    use_wc, use_3x, use_nn = ("wildcard" in chips_avail, "3x_booster" in chips_avail,
                              "no_negative" in chips_avail)
    first_free = state.prev_team is None
    BIG = float(E)

    # ---- variable layout -------------------------------------------------
    idx = {}
    n = 0

    def var(name, size=1):
        nonlocal n
        idx[name] = np.arange(n, n + size)
        n += size

    for t in range(T):
        var(("x", t), E)
        var(("b1", t), E)         # x2 boost (driver entries used)
        var(("b3", t), E)         # x3 boost via chip
        var(("wc", t))
        var(("nn", t))
        var(("w", t), E)          # nn & x
        var(("bn", t), E)         # nn & b1
        var(("in", t), E)         # transfers in
        var(("u", t))             # free transfers used (value, before wildcard waiver)
        var(("s", t))             # 1 if n_tr <= ft
        var(("ex", t))            # paid transfers
        var(("ug", t))            # free transfers actually consumed
        var(("ft", t))            # free transfers available
        var(("W", t))             # wealth
    lb, ub = np.zeros(n), np.ones(n) * np.inf
    integrality = np.zeros(n)
    c = np.zeros(n)

    def setb(name, lo=None, hi=None, integer=False):
        ii = idx[name]
        if lo is not None:
            lb[ii] = lo
        if hi is not None:
            ub[ii] = hi
        if integer:
            integrality[ii] = 1

    for t in range(T):
        for nm in ("x", "b1", "b3", "wc", "nn", "w", "bn", "s"):
            setb((nm, t), 0, 1, True)
        setb(("in", t), 0, 1)
        setb(("u", t), 0, rules.max_bank + E, True)
        setb(("ex", t), 0, E)
        setb(("ug", t), 0, rules.max_bank + E)
        setb(("ft", t), 0, rules.max_bank, True)
        setb(("W", t), 0, 1e6)
        # variables that cannot exist
        ub[idx[("b1", t)][~is_driver]] = 0
        ub[idx[("b3", t)][~is_driver]] = 0
        ub[idx[("bn", t)][~is_driver]] = 0
        if not use_3x:
            ub[idx[("b3", t)]] = 0
        if not use_wc:
            ub[idx[("wc", t)]] = 0
        if not use_nn:
            ub[idx[("nn", t)]] = 0
            ub[idx[("w", t)]] = 0
            ub[idx[("bn", t)]] = 0

    # ---- objective (milp minimises, so negate) ---------------------------
    for t in range(T):
        d = disc[t]
        c[idx[("x", t)]] -= d * pts[t]
        c[idx[("b1", t)]] -= d * pts[t] * is_driver
        c[idx[("b3", t)]] -= d * 2.0 * pts[t] * is_driver
        g = np.maximum(-pts[t], 0) if nn_gain is None else np.asarray(nn_gain[t], float)
        c[idx[("w", t)]] -= d * g
        c[idx[("bn", t)]] -= d * g * is_driver
        c[idx[("ex", t)]] += d * rules.penalty
        if value_weight:
            nxt = price[t + 1] if t + 1 < T else (terminal_price if terminal_price is not None else price[t])
            c[idx[("x", t)]] -= d * value_weight * (np.asarray(nxt, float) - price[t])
        if chip_cost:
            c[idx[("wc", t)]] += d * chip_cost.get("wildcard", 0.0)
            c[idx[("nn", t)]] += d * chip_cost.get("no_negative", 0.0)
            c[idx[("b3", t)]] += d * chip_cost.get("3x_booster", 0.0) * is_driver

    rows, lo_, hi_ = [], [], []

    def add(coefs, lo, hi):
        rows.append(coefs)
        lo_.append(lo)
        hi_.append(hi)

    for t in range(T):
        x, b1, b3 = idx[("x", t)], idx[("b1", t)], idx[("b3", t)]
        wc, nn = idx[("wc", t)][0], idx[("nn", t)][0]
        # team composition
        add({**{i: 1 for i in x[is_driver]}}, rules.n_drivers, rules.n_drivers)
        add({**{i: 1 for i in x[~is_driver]}}, rules.n_constructors, rules.n_constructors)
        # exactly one boost, on a held driver
        add({**{i: 1 for i in b1[is_driver]}, **{i: 1 for i in b3[is_driver]}}, 1, 1)
        for e in D:
            add({b1[e]: 1, b3[e]: 1, x[e]: -1}, -np.inf, 0)
        # one chip per weekend
        add({wc: 1, nn: 1, **{i: 1 for i in b3[is_driver]}}, 0, 1)
        # no-negative linking (upper bounds suffice: gains are non-negative)
        for e in range(E):
            add({idx[("w", t)][e]: 1, nn: -1}, -np.inf, 0)
            add({idx[("w", t)][e]: 1, x[e]: -1}, -np.inf, 0)
        for e in D:
            add({idx[("bn", t)][e]: 1, nn: -1}, -np.inf, 0)
            add({idx[("bn", t)][e]: 1, b1[e]: -1}, -np.inf, 0)
        # budget
        W = idx[("W", t)][0]
        add({**{x[e]: price[t, e] for e in range(E)}, W: -1}, -np.inf, 0)
        # wealth dynamics
        if t + 1 < T:
            Wn = idx[("W", t + 1)][0]
            add({Wn: 1, W: -1, **{x[e]: -(price[t + 1, e] - price[t, e]) for e in range(E)}}, 0, 0)

        # transfers
        inn, u, s, ex, ug, ft = (idx[("in", t)], idx[("u", t)][0], idx[("s", t)][0],
                                 idx[("ex", t)][0], idx[("ug", t)][0], idx[("ft", t)][0])
        if t == 0 and first_free:
            ub[inn] = 0
            ub[[u, ex, ug]] = 0
            ub[s] = 0
        else:
            xprev = None if t > 0 else None
            for e in range(E):
                if t == 0:
                    pv = 1.0 if e in state.prev_team else 0.0
                    add({inn[e]: 1, x[e]: -1}, -pv, np.inf)
                else:
                    add({inn[e]: 1, x[e]: -1, idx[("x", t - 1)][e]: 1}, 0, np.inf)
            ntr = {i: 1 for i in inn}
            # u = min(n_tr, ft) via big-M
            add({u: 1, **{i: -1 for i in inn}}, -np.inf, 0)
            add({u: 1, ft: -1}, -np.inf, 0)
            add({u: 1, **{i: -1 for i in inn}, s: -BIG}, -BIG, np.inf)       # u >= n - M(1-s)
            add({u: 1, ft: -1, s: BIG}, 0, np.inf)                           # u >= ft - M*s
            # paid transfers
            add({ex: 1, **{i: -1 for i in inn}, u: 1, idx[("wc", t)][0]: BIG}, 0, np.inf)
            # consumed free transfers (wildcard waives)
            add({ug: 1, u: -1, idx[("wc", t)][0]: BIG}, 0, np.inf)
        # free-transfer bank
        if t == 0:
            lb[ft] = ub[ft] = min(state.free_transfers, rules.max_bank)
        if t + 1 < T:
            ftn = idx[("ft", t + 1)][0]
            add({ftn: 1, ft: -1, ug: 1}, -np.inf, rules.free_per_round)
            if t == 0 and first_free:
                add({ftn: 1}, -np.inf, min(rules.max_bank, state.free_transfers + rules.free_per_round))

    for t in frozen:                       # team parked: no transfers, no chips this round
        if t == 0 and first_free:
            raise ValueError("cannot freeze the first round of a fresh team")
        ub[idx[("in", t)]] = 0
        ub[[idx[("wc", t)][0], idx[("nn", t)][0]]] = 0
        ub[idx[("b3", t)]] = 0
    # wealth at round 0
    W0 = state.wealth if state.wealth is not None else rules.budget
    ub[idx[("W", 0)]] = W0
    lb[idx[("W", 0)]] = W0
    # chips used at most once
    if use_wc:
        add({idx[("wc", t)][0]: 1 for t in range(T)}, 0, 1)
    if use_nn:
        add({idx[("nn", t)][0]: 1 for t in range(T)}, 0, 1)
    if use_3x:
        add({i: 1 for t in range(T) for i in idx[("b3", t)][is_driver]}, 0, 1)

    A = lil_matrix((len(rows), n))
    for r, coefs in enumerate(rows):
        for j, v in coefs.items():
            A[r, j] += v
    res = milp(c, constraints=LinearConstraint(A.tocsr(), lo_, hi_), integrality=integrality,
               bounds=Bounds(lb, ub),
               options={"time_limit": time_limit, "mip_rel_gap": mip_rel_gap, "disp": verbose})
    if res.x is None:
        return Plan([], [], [], [], [], [], -np.inf, status=res.message, optimal=False)
    xs = res.x
    teams, boosts, chips, tr_in, pens, rpts = [], [], [], [], [], []
    for t in range(T):
        teams.append(frozenset(int(e) for e in np.flatnonzero(xs[idx[("x", t)]] > 0.5)))
        b = np.flatnonzero((xs[idx[("b1", t)]] + xs[idx[("b3", t)]]) > 0.5)
        boosts.append(int(b[0]))
        chip = None
        if xs[idx[("wc", t)][0]] > 0.5:
            chip = "wildcard"
        elif xs[idx[("nn", t)][0]] > 0.5:
            chip = "no_negative"
        elif xs[idx[("b3", t)]].max() > 0.5:
            chip = "3x_booster"
        chips.append(chip)
    total, per, pen, trs, ok = evaluate_plan(teams, boosts, chips, pts, price, is_driver, rules, state, discount)
    if nn_gain is not None:                  # expected value of No Negative under uncertainty, not just on the means
        for t in range(T):
            if chips[t] == "no_negative":
                extra = sum(nn_gain[t][e] - max(-pts[t][e], 0.0) for e in teams[t]) + nn_gain[t][boosts[t]] - max(-pts[t][boosts[t]], 0.0)
                per[t] += float(extra)
                total += float(extra) * (1.0 if discount is None else discount[t])
    return Plan(teams, boosts, chips, trs, pen, per, total, objective=-res.fun,
                status=res.message, optimal=bool(res.status == 0))


# ──────────────────────────────────────────────────────────────
# True exhaustive enumeration (tiny instances only) - used to validate the MILP
# ──────────────────────────────────────────────────────────────

def brute_force_strategy(pts, price, is_driver, rules=None, state=None):
    """Enumerate every team sequence x boost x chip assignment. Exponential: tests only."""
    pts = np.asarray(pts, float)
    price = np.asarray(price, float)
    is_driver = np.asarray(is_driver, bool)
    rules = rules or Rules()
    state = state or State(free_transfers=rules.free_per_round, chips_left=rules.chips)
    T, E = pts.shape
    D = [e for e in range(E) if is_driver[e]]
    C = [e for e in range(E) if not is_driver[e]]
    teams = [frozenset(d + c) for d in combinations(D, rules.n_drivers)
             for c in combinations(C, rules.n_constructors)]
    chip_opts = [None] + [c for c in rules.chips if c in state.chips_left]
    best = (-np.inf, None)

    def rec(t, prev, W, ft, used, seq):
        nonlocal best
        if t == T:
            tot = sum(r for r, *_ in seq)
            if tot > best[0] + 1e-9:
                best = (tot, list(seq))
            return
        for team in teams:
            cost = sum(price[t][e] for e in team)
            if cost > W + 1e-9:
                continue
            n_in = 0 if prev is None else len(team - prev)
            for chip in chip_opts:
                if chip is not None and chip in used:
                    continue
                pen = 0.0 if (prev is None or chip == "wildcard") else rules.penalty * max(0, n_in - ft)
                usedft = 0 if (prev is None or chip == "wildcard") else min(n_in, ft)
                ft2 = min(rules.max_bank, ft - usedft + rules.free_per_round)
                v = _chip_pts(np.array([pts[t][e] for e in sorted(team)]), chip).sum()
                for b in [e for e in team if is_driver[e]]:
                    bp = _chip_pts(np.array([pts[t][b]]), chip)[0]
                    r = v + (2.0 if chip == "3x_booster" else 1.0) * bp - pen
                    W2 = W + (sum(price[t + 1][e] - price[t][e] for e in team) if t + 1 < T else 0)
                    rec(t + 1, team, W2, ft2, used | ({chip} if chip else set()),
                        seq + [(r, team, b, chip)])

    rec(0, state.prev_team, state.wealth if state.wealth is not None else rules.budget,
        state.free_transfers, set(), [])
    return best
