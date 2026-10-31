#!/usr/bin/env python3
"""
The best possible 2026 season so far, with hindsight (season_optimum.py)
========================================================================
Every decision from the first gameday to the last completed one that would have scored the most OFFICIAL points
(the game's own feed: fantasy_f1_official_points.csv / fantasy_f1_prices_official.csv), under the game's rules:

  * $100m at the start; the game's real price of every asset each week (your team value moves with them)
  * 5 drivers + 2 constructors, a 2x boost every week
  * 2 free transfers a race (one unused carries over, max 3), -10 per extra transfer
  * one chip per race, each once: Wildcard, x3 Boost (x3 on a second driver, the 2x stays), No Negative,
    Limitless (that week any team regardless of budget; your team comes back afterwards and no free transfers
    carry over out of that week)

Exact: the solver (f1_strategy.solve_strategy, branch-and-bound) proves each plan optimal; Limitless is placed by
trying every week (those solves run in parallel).  The output says if any solve hit its time limit unproven.

Not included:
  Autopilot   with hindsight the 2x is already on the week's top scorer, so it cannot add anything
  Final Fix   a swap after qualifying needs each driver's race-only points, which the CSVs do not hold
  stand-in assets (e.g. Lawson for Red Bull in rounds 14-16) are not in the model's asset list

    python fetch_fantasy_feed.py                 # latest official points and prices
    python season_optimum.py                     # table + results/season_2026_hindsight_optimum.csv
    python season_optimum.py --upto 10 --time-limit 300 --workers 4 --no-limitless
"""
import argparse
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

import f1_planner as FP
from f1_calendar import RACE_CALENDAR_2026
from f1_strategy import Rules, State, evaluate_plan, solve_strategy

CHIPS = ("wildcard", "3x_booster", "no_negative")
OUT_CSV = os.path.join("results", "season_2026_hindsight_optimum.csv")


def load_official(points_csv="fantasy_f1_official_points.csv", prices_csv="fantasy_f1_prices_official.csv", upto=None):
    """(gamedays, race names, points G x E, prices G x E) for every completed gameday, in FP.ENTITIES order."""
    pts, pr = pd.read_csv(points_csv), pd.read_csv(prices_csv)
    gds = sorted(int(g) for g in pts["gameday"].unique() if upto is None or int(g) <= upto)
    E = len(FP.ENTITIES)
    col = {}
    for n in set(pts["name"]) | set(pr["name"]):
        c = FP.canon_entity(n)
        if c in FP.ENTITIES:
            col[n] = FP.ENTITIES.index(c)
    P = np.zeros((len(gds), E))                      # an asset that did not take part scores 0
    C = np.full((len(gds), E), np.nan)
    for i, g in enumerate(gds):
        for r in pts[pts["gameday"] == g].itertuples():
            if r.name in col:
                P[i, col[r.name]] = float(r.points)
        for r in pr[pr["gameday"] == g].itertuples():
            if r.name in col:
                C[i, col[r.name]] = float(r.price)
        if np.isnan(C[i]).any():                     # missing from a week's price list: keeps its previous price
            if i == 0:
                missing = [FP.ENTITIES[e] for e in np.flatnonzero(np.isnan(C[i]))]
                raise ValueError(f"no gameday-{g} price for {missing}")
            C[i] = np.where(np.isnan(C[i]), C[i - 1], C[i])
    rounds = [int(pts.loc[pts["gameday"] == g, "round"].iloc[0]) for g in gds]
    names = [next((r["name"] for r in RACE_CALENDAR_2026 if r["round"] == rnd), f"round {rnd}") for rnd in rounds]
    return gds, names, P, C


def _solve(args):
    """One season solve; k = the Limitless week (None: Limitless not used)."""
    P, C, k, chips, time_limit, gap = args
    rules = Rules(chips=chips)
    state = State(prev_team=None, wealth=rules.budget, free_transfers=rules.free_per_round, chips_left=chips)
    t0 = time.time()
    if k is None:
        plan = solve_strategy(P, C, FP.IS_DRIVER, rules, state, time_limit=time_limit, mip_rel_gap=gap)
        return k, plan, None, time.time() - t0
    lt, lb, lv = FP.limitless_team(P[k])
    Pk = np.array(P, float)
    Pk[k] = 0.0                                      # the parked real team does not score that week
    plan = solve_strategy(Pk, C, FP.IS_DRIVER, rules, state, time_limit=time_limit, mip_rel_gap=gap, frozen=(k,))
    return k, plan, (lt, lb, lv), time.time() - t0


def _line(k, plan, lim, secs):
    extra = lim[2] if lim else 0.0
    proof = "proven" if plan.optimal else ("NOT proven" if plan.bound is None else f"at most {plan.bound + extra:.0f}")
    return f"  {'no Limitless' if k is None else f'Limitless in week {k + 1}':<22s} {plan.total + extra:8.1f}  {proof:<16s} {secs:5.0f}s"


def optimise(P, C, limitless=True, time_limit=1200, gap=0.0, workers=None, chips=CHIPS, log=print):
    """Best season. Returns dict(plan, k, lim, total, base_total, runs) - k is the Limitless week or None."""
    G = len(P)
    jobs = [None] + (list(range(1, G)) if limitless else [])     # week 0 is the team's first pick, nothing to park
    args = [(P, C, k, tuple(chips), time_limit, gap) for k in jobs]
    runs = {}
    if workers == 1 or len(args) == 1:
        it = map(_solve, args)
        for k, plan, lim, secs in it:
            runs[k] = (plan, lim, secs)
            log(_line(k, plan, lim, secs))
    else:
        # "spawn": HiGHS leaves threads behind, and forked workers can inherit their locks and hang
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
            futs = [ex.submit(_solve, a) for a in args]
            for f in as_completed(futs):
                k, plan, lim, secs = f.result()
                runs[k] = (plan, lim, secs)
                log(_line(k, plan, lim, secs))
    best_k = max(runs, key=lambda k: runs[k][0].total + (runs[k][1][2] if runs[k][1] else 0.0))
    plan, lim, _ = runs[best_k]
    total = plan.total + (lim[2] if lim else 0.0)
    # the best any season could score: each solve's proven bound (its total when it finished)
    ceiling = max((p.total if p.optimal else np.inf if p.bound is None else p.bound) + (lm[2] if lm else 0.0)
                  for p, lm, _ in runs.values())
    return {"plan": plan, "k": best_k, "lim": lim, "total": total, "base_total": runs[None][0].total, "runs": runs,
            "ceiling": max(ceiling, total)}


def season_table(best, gds, names, P, C):
    """Race-by-race decisions of the best season (real team, Limitless team, transfers, boosts, bank)."""
    plan, k, lim = best["plan"], best["k"], best["lim"]
    rules = Rules()
    rows, prev, wealth, cum = [], None, rules.budget, 0.0
    for t in range(len(gds)):
        real = plan.teams[t]                         # in the Limitless week this is the parked team
        if t == k:
            team, boost, b3, chip, points = lim[0], lim[1], None, "limitless", lim[2]
        else:
            team, boost, b3, chip, points = real, plan.boosts[t], (plan.boosts3 or [None] * len(gds))[t], plan.chips[t], plan.round_points[t]
        cost = float(C[t][list(real)].sum())
        cum += points
        rows.append({
            "gameday": gds[t], "race": names[t], "chip": chip or "",
            "transfers": plan.transfers_in[t], "penalty": plan.penalties[t],
            "out": "" if prev is None or t == k else ", ".join(sorted(FP.ENTITIES[e] for e in (prev - team))),
            "in": "(one-race Limitless team; yours returns next race)" if t == k else
                  "" if prev is None else ", ".join(sorted(FP.ENTITIES[e] for e in (team - prev))),
            "boost_2x": FP.ENTITIES[boost], "boost_3x": FP.ENTITIES[b3] if b3 is not None else "",
            "drivers": ", ".join(FP.ENTITIES[e] for e in sorted(team) if FP.IS_DRIVER[e]),
            "constructors": ", ".join(FP.ENTITIES[e] for e in sorted(team) if not FP.IS_DRIVER[e]),
            "points": round(points, 1), "cumulative": round(cum, 1),
            "team_value": round(cost, 1), "bank": round(wealth - cost, 1) + 0.0,
        })
        if t + 1 < len(gds):
            wealth += float((C[t + 1] - C[t])[list(real)].sum())
        prev = real                                  # transfers are counted from the real team
    return pd.DataFrame(rows)


def check(best, P, C):
    """Re-score the plan independently with the rules' single definition (evaluate_plan)."""
    plan, k, lim = best["plan"], best["k"], best["lim"]
    rules = Rules(chips=CHIPS)
    state = State(prev_team=None, wealth=rules.budget, free_transfers=rules.free_per_round, chips_left=CHIPS)
    Pk = np.array(P, float)
    frozen = ()
    if k is not None:
        Pk[k] = 0.0
        frozen = (k,)
    tot, per, pen, trs, ok = evaluate_plan(plan.teams, plan.boosts, plan.chips, Pk, C, FP.IS_DRIVER, rules, state,
                                           frozen=frozen, plan_boosts3=plan.boosts3)
    return ok, tot + (lim[2] if lim else 0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--upto", type=int, help="last gameday to include (default: every completed one)")
    ap.add_argument("--no-limitless", action="store_true", help="only Wildcard, x3 Boost and No Negative")
    ap.add_argument("--time-limit", type=int, default=1200, help="seconds per solve (default 1200)")
    ap.add_argument("--gap", type=float, default=0.0, help="accepted optimality gap, e.g. 0.001 (default: exact)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--csv", default=OUT_CSV)
    a = ap.parse_args()

    gds, names, P, C = load_official(upto=a.upto)
    print(f"Official points and prices for gamedays {gds[0]}-{gds[-1]} ({len(gds)} races): {names[0]} .. {names[-1]}")
    n_solves = 1 + (0 if a.no_limitless else len(gds) - 1)
    print(f"Solving {n_solves} season plan(s) on {a.workers} worker(s), up to {a.time_limit}s each:")
    best = optimise(P, C, limitless=not a.no_limitless, time_limit=a.time_limit, gap=a.gap, workers=a.workers)
    df = season_table(best, gds, names, P, C)
    ok, rescored = check(best, P, C)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)
    print("\nBEST POSSIBLE SEASON (hindsight)\n")
    print(df[["gameday", "race", "chip", "transfers", "penalty", "out", "in", "boost_2x", "boost_3x", "points",
              "cumulative", "bank"]].to_string(index=False))
    print("\nTeams:")
    for r in df.itertuples():
        print(f"  {r.gameday:2d} {r.race:16s} {r.drivers} | {r.constructors}")
    unproven = [k for k, (p, _, _) in best["runs"].items() if not p.optimal]
    room = best["ceiling"] - best["total"]
    print(f"\nTotal: {best['total']:.0f} points"
          + (f"  (Limitless in {names[best['k']]} is worth {best['total'] - best['base_total']:+.0f} over the best "
             f"season without it, {best['base_total']:.0f})" if best["k"] is not None else ""))
    print(f"Independent re-score: {rescored:.0f} ({'matches' if ok and abs(rescored - best['total']) < 1e-6 else 'MISMATCH'})")
    if not unproven:
        print("Every solve proved optimal: no season under these rules scores more.")
    elif room < 1.0:                                 # official points are whole numbers
        print(f"Proven: {len(unproven)} solve(s) stopped at the time limit, but their bounds show none can beat this.")
    else:
        print(f"Not proven: {len(unproven)} solve(s) stopped at the time limit; the best season is at most "
              f"{room:.0f} points above this - re-run with a larger --time-limit to settle it.")
    os.makedirs(os.path.dirname(a.csv) or ".", exist_ok=True)
    df.to_csv(a.csv, index=False)
    print(f"Wrote {a.csv}")


if __name__ == "__main__":
    main()
