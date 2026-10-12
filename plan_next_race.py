#!/usr/bin/env python3
"""
Best transfers for the next race (plan_next_race.py)
====================================================
Forecasts the next few raced weekends from all results so far, then solves exactly (f1_strategy MILP) for the
transfers / boost that maximise expected points over that horizon, from YOUR current team, bank and free transfers.

    python plan_next_race.py --team Hamilton Antonelli Colapinto Lawson Bearman Ferrari "Racing Bulls" \\
                             --bank 2.4 --free 3 [--chips wildcard 3x_booster no_negative] [--horizon 3]

Prices: the game's current prices (fantasy_f1_prices_official.csv from fetch_fantasy_feed.py), held flat
over the horizon. Forecast: the weekend simulator incl. Driver of the Day, optionally shrunk toward an EWMA of
the game's official points (--ewma, 0.3 was the strongest lever in the 2026 policy search; in-sample).
Rules (2 free transfers per race, bank up to 3, -10 per extra) are this project's assumptions.
"""
import argparse

import numpy as np
import pandas as pd

import f1_season as S
from f1_calendar import RACE_CALENDAR_2026
from f1_strategy import Rules, State, evaluate_plan, solve_strategy
from f1_weekend_data import load_results
from f1_weekend_model import WeekendForecaster


def forecasts(res, rounds, n=8000):
    f = WeekendForecaster(res, year=2026, verbose=False)
    out = []
    for r in rounds:
        fc = f.forecast(r, n=n, seed=100 + r)
        d = dict(zip(fc.drivers, fc.total_points(include_dotd=True).mean(0)))
        c = dict(zip(fc.teams, fc.constructor_total().mean(0)))
        out.append(np.array([d.get(e, 0.0) for e in S.DRIVERS] + [c.get(e, 0.0) for e in S.TEAMS]))
    return np.array(out)


def ewma_official(path, alpha=0.2):
    op = pd.read_csv(path)
    w = op.pivot_table(index="gameday", columns="name", values="points").reindex(columns=S.ENTITIES).fillna(0.0)
    e = w.iloc[0].values.astype(float)
    for _, row in w.iloc[1:].iterrows():
        e = alpha * row.values + (1 - alpha) * e
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", nargs=7, required=True)
    ap.add_argument("--bank", type=float, required=True, help="cost cap remaining ($m)")
    ap.add_argument("--free", type=int, required=True, help="free transfers available")
    ap.add_argument("--chips", nargs="*", default=[], help="chips still available: wildcard 3x_booster no_negative")
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--discount", type=float, default=0.6)
    ap.add_argument("--ewma", type=float, default=0.0)
    ap.add_argument("--prices", default="fantasy_f1_prices_official.csv")
    ap.add_argument("--official", default="fantasy_f1_official_points.csv")
    a = ap.parse_args()

    res = load_results(verbose=False)
    done = set(res.weekends[res.weekends["year"] == 2026]["round"])
    upcoming = [r["round"] for r in RACE_CALENDAR_2026 if not r.get("cancelled") and r["round"] not in done][:a.horizon]
    pts = forecasts(res, upcoming)
    if a.ewma:
        pts = (1 - a.ewma) * pts + a.ewma * ewma_official(a.official)[None, :]
    pr = pd.read_csv(a.prices)
    cur = pr[pr["gameday"] == pr["gameday"].max()].set_index("name")["price"]
    price = np.array([float(cur[e]) for e in S.ENTITIES])
    team = frozenset(S.ENTITIES.index(S._canon_entity(n)) for n in a.team)
    wealth = float(price[list(team)].sum()) + a.bank
    rules = Rules()
    state = State(prev_team=team, wealth=wealth, free_transfers=a.free, chips_left=tuple(a.chips))
    H = len(upcoming)
    disc = a.discount ** np.arange(H)
    plan = solve_strategy(pts, np.repeat(price[None, :], H, 0), S.IS_DRIVER, rules, state, disc, time_limit=120, mip_rel_gap=0.0)

    names = lambda s: sorted(S.ENTITIES[e] for e in s)
    def race1(t, boost, chip):
        tot, per, pen, trs, ok = evaluate_plan([t], [boost], [chip], pts[:1], price[None, :], S.IS_DRIVER, rules, state)
        return per[0], pen[0], trs[0], ok
    best_keep_boost = max((e for e in team if S.IS_DRIVER[e]), key=lambda e: pts[0][e])
    keep = race1(team, best_keep_boost, None)
    t0, b0, c0 = plan.teams[0], plan.boosts[0], plan.chips[0]
    new = race1(t0, b0, c0)
    print(f"Next races: {[next(r['name'] for r in RACE_CALENDAR_2026 if r['round'] == x) for x in upcoming]}  "
          f"(team value + bank = ${wealth:.1f}m, {a.free} free transfers, chips {a.chips or 'none given'})")
    print(f"\nKeep the team, boost {S.ENTITIES[best_keep_boost]}: expected {keep[0]:.1f} pts in {RACE_CALENDAR_2026[upcoming[0] - 1]['name']}")
    print(f"Recommended:  OUT {sorted(set(names(team)) - set(names(t0)))}  IN {sorted(set(names(t0)) - set(names(team)))}"
          f"  boost {S.ENTITIES[b0]}  chip {c0 or '-'}  ->  expected {new[0]:.1f} pts (transfer penalty {new[1]:.0f})")
    print(f"Horizon plan (optimal: {plan.optimal}):")
    for k in range(H):
        print(f"  {RACE_CALENDAR_2026[upcoming[k] - 1]['name']:16s} team {names(plan.teams[k])}  boost {S.ENTITIES[plan.boosts[k]]}"
              f"  chip {plan.chips[k] or '-'}  transfers {plan.transfers_in[k]}  exp {plan.round_points[k]:.1f}")
    print("\nForecast for the next race (expected points; per-driver error is typically +-11):")
    order = np.argsort(-pts[0])
    rows = [(S.ENTITIES[e], "D" if S.IS_DRIVER[e] else "C", round(pts[0][e], 1), price[e], round(pts[0][e] / price[e], 2),
             "*" if e in team else "") for e in order]
    print(pd.DataFrame(rows, columns=["asset", "type", "exp_pts", "price", "pts_per_$m", "yours"]).to_string(index=False))


if __name__ == "__main__":
    main()
