#!/usr/bin/env python3
"""
Best changes for the next race (plan_next_race.py)
==================================================
From your current team, bank, free transfers and chips: the exact best transfers, boost and chip for the next race,
planned over the next few races so this week's moves also suit the following ones (f1_planner, same model as
plan_season.py).  Re-run after practice (fetch_openf1_full.py first) to use that weekend's practice pace.

    python plan_next_race.py --team Hamilton Antonelli Colapinto Lawson Bearman Ferrari "Racing Bulls" \\
                             --bank 2.4 --free 3 [--chips 3x_booster limitless] [--horizon 3] [--keep ...] [--avoid ...]
"""
import argparse

import numpy as np
import pandas as pd

import f1_planner as FP
from plan_season import forecasts, load_specs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--team", nargs=7, required=True)
    ap.add_argument("--bank", type=float, required=True, help="cost cap remaining ($m)")
    ap.add_argument("--free", type=int, required=True, help="free transfers available")
    ap.add_argument("--chips", nargs="*", default=[], help="chips still available: wildcard 3x_booster no_negative limitless")
    ap.add_argument("--horizon", type=int, default=3, help="races planned ahead (the first one is acted on)")
    ap.add_argument("--discount", type=float, default=0.6, help="weight of each later race (0.6 was best in the 2026 replay)")
    ap.add_argument("--price", action="append", default=[], help="override a price, e.g. --price Antonelli=26.5")
    ap.add_argument("--keep", nargs="*", default=[])
    ap.add_argument("--avoid", nargs="*", default=[])
    ap.add_argument("--spec", default=FP.SPEC_FILE)
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()

    import mlflow_experiments as X
    res, truth, _ = X.get_data()
    rounds = FP.upcoming_rounds(res, a.horizon)
    names = [next(r["name"] for r in FP.RACE_CALENDAR_2026 if r["round"] == x) for x in rounds]
    specs, c_sim, spec_name = load_specs(a.spec)
    pts = forecasts(rounds, specs, c_sim, a.refresh)
    price_now = FP.current_prices(overrides=dict(p.split("=", 1) for p in a.price))
    path = FP.project_prices(price_now, pts, FP.official_tables())
    team = [FP.canon_entity(n) for n in a.team]
    plan, state = FP.plan(pts, path, team=team, bank=a.bank, free=a.free, chips=a.chips, discount=a.discount,
                          locked_in=a.keep, locked_out=a.avoid)
    rules = FP.Rules()
    keep_boost = max((FP.ENTITIES.index(n) for n in team if n in FP.DRIVERS), key=lambda e: pts[0][e])
    idx = frozenset(FP.ENTITIES.index(n) for n in team)
    keep_tot, keep_per, *_ = FP.evaluate_plan([idx], [keep_boost], [None], pts[:1], path[:1], FP.IS_DRIVER, rules, state)
    t0 = [FP.ENTITIES[e] for e in sorted(plan.teams[0])]
    print(f"Model: {spec_name}\nNext race: {names[0]} | value + bank ${state.wealth:.1f}m | {a.free} free transfer(s) | chips {a.chips or 'none'}")
    if plan.chips[0] == "limitless":
        print(f"\nPLAY LIMITLESS this week: team {t0}, boost {FP.ENTITIES[plan.boosts[0]]} (your team returns afterwards)")
    else:
        print(f"\nKeep the team (boost {FP.ENTITIES[keep_boost]}): expected {keep_per[0]:.1f} pts")
        print(f"Recommended:  OUT {sorted(set(team) - set(t0)) or '-'}  IN {sorted(set(t0) - set(team)) or '-'}  "
              f"boost {FP.ENTITIES[plan.boosts[0]]}  chip {plan.chips[0] or '-'}  -> expected {plan.round_points[0]:.1f} pts "
              f"(transfer penalty {plan.penalties[0]:.0f})")
    print(f"\nPlan for the next {len(rounds)} races (only the first is acted on; re-run each week; optimal: {plan.optimal}):")
    prev = set(team)
    for k, nm in enumerate(names):
        tk = [FP.ENTITIES[e] for e in sorted(plan.teams[k])]
        print(f"  {nm:16s} {', '.join(tk)} | boost {FP.ENTITIES[plan.boosts[k]]} | chip {plan.chips[k] or '-'} | "
              f"in {sorted(set(tk) - prev) or '-'} | exp {plan.round_points[k]:.1f}")
        if plan.chips[k] != "limitless":
            prev = set(tk)
    nxt = pd.DataFrame({"asset": FP.ENTITIES, "type": np.where(FP.IS_DRIVER, "D", "C"), "exp_points": pts[0].round(1),
                        "price": price_now, "pts_per_$m": (pts[0] / price_now).round(2),
                        "yours": ["*" if e in team else "" for e in FP.ENTITIES]}).sort_values("exp_points", ascending=False)
    print(f"\nForecast for {names[0]} (typical error +-11-15 points per driver):\n" + nxt.to_string(index=False))


if __name__ == "__main__":
    main()
