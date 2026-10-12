#!/usr/bin/env python3
"""
Best lineups for the rest of the season (plan_season.py)
========================================================
Forecasts every remaining 2026 race with the selected model (f1_planner) and solves exactly for the team, boost,
transfers and chips of each race from YOUR starting point.

    # your current team
    python plan_season.py --team Hamilton Antonelli Colapinto Lawson Bearman Ferrari "Racing Bulls" --bank 2.4 --free 3 \\
                          --chips 3x_booster no_negative
    # a fresh team with a $100m budget, or any prices you want to try
    python plan_season.py --budget 100 --price Antonelli=26.5 --price Mercedes=33.0
    # constraints
    python plan_season.py ... --keep Antonelli --avoid Stroll --rounds 3

Prices: the game's current prices (fetch_fantasy_feed.py), projected forward with the game's price rule
(--flat-prices to hold them).  Rules: 2 free transfers per race, bank up to 3, -10 per extra transfer (assumed);
chips modelled: wildcard, 3x_booster, no_negative, limitless (Final Fix and Autopilot are NOT modelled).
Forecasts are cached in results/cache/ (delete it after new results come in; fetch_fantasy_feed.py first).
"""
import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd

import f1_planner as FP

DEFAULT_SPEC = {"family": "xgb", "hp": {"n_estimators": 618, "max_depth": 1, "learning_rate": 0.0096, "min_child_weight": 41.6,
                                       "subsample": 0.5, "colsample_bytree": 0.42, "reg_lambda": 0.107},
                "fset": "all", "half_life": 25, "w26": 3.0, "use_official": True, "ewma_alpha": 0.3,
                "b_early": 0.8, "b_late": 0.3, "kappa": 4.0, "c_ewma": 0.1}


def load_specs(path):
    if path and os.path.exists(path):
        d = FP.load_spec(path)
        return d["specs"], (d.get("c_sim", 0.5), d.get("bonus_alpha", 0.3), d.get("planning_blend")), d.get("name", os.path.basename(path))
    return [DEFAULT_SPEC], (0.5, 0.3, None), "built-in default (no results/s26_final_spec.json yet)"


def forecasts(rounds, specs, c_sim, refresh=False):
    """Expected points (len(rounds) x 33) - computed once for ALL remaining rounds and cached, then sliced."""
    import mlflow_experiments as X
    res, _, _ = X.get_data()
    allr = FP.upcoming_rounds(res)
    full_pts = _forecasts_all(allr, specs, c_sim, refresh)
    return full_pts[[allr.index(r) for r in rounds]]


def _forecasts_all(rounds, specs, c_sim, refresh=False):
    import mlflow_experiments as X, f1_practice as P, openf1_cache as C
    from experiments import zoo, s26
    res, truth, ctruth = X.get_data()
    key = hashlib.sha1(json.dumps([X.data_fingerprint(res), specs, c_sim, rounds,
                                   os.path.getmtime("fantasy_f1_official_points.csv")], default=str).encode()).hexdigest()[:16]
    os.makedirs(os.path.join("results", "cache"), exist_ok=True)
    path = os.path.join("results", "cache", f"forecast_{key}.pkl")
    if os.path.exists(path) and not refresh:
        return pd.read_pickle(path)
    full = C.load_all(only_complete=False)              # includes a weekend in progress (its practice, before the lock)
    pdata = C.to_practice(full)
    tracks = zoo.track_tables()
    panel_pkl = os.path.join("results", "cache", f"panel_{X.data_fingerprint(res)}.pkl")
    if not os.path.exists(panel_pkl):
        zoo.build_panel(res, truth, full=full, pdata=pdata, tracks=tracks).to_pickle(panel_pkl)
    D = s26.prepare(panel_pkl, os.path.join("results", "cache", f"s26_{X.data_fingerprint(res)}.pkl"))
    pts = FP.forecast_remaining(res, truth, specs, rounds, D["panel"], full=full, pdata=pdata, tracks=tracks,
                                c_sim=c_sim[0], bonus_alpha=c_sim[1], planning_blend=c_sim[2])
    pd.to_pickle(pts, path)
    return pts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--team", nargs="*", default=None, help="your 5 drivers + 2 constructors (omit for a fresh team)")
    ap.add_argument("--bank", type=float, default=0.0, help="cost cap remaining ($m), with --team")
    ap.add_argument("--budget", type=float, default=100.0, help="budget for a fresh team ($m)")
    ap.add_argument("--free", type=int, default=2, help="free transfers available now")
    ap.add_argument("--chips", nargs="*", default=[], help="chips still available: wildcard 3x_booster no_negative limitless autopilot (autopilot: best week reported)")
    ap.add_argument("--rounds", type=int, default=None, help="plan only the next N races")
    ap.add_argument("--discount", type=float, default=0.9, help="weight of each later race (forecasts get less reliable)")
    ap.add_argument("--price", action="append", default=[], help="override a price, e.g. --price Antonelli=26.5")
    ap.add_argument("--flat-prices", action="store_true", help="hold prices flat instead of projecting price changes")
    ap.add_argument("--keep", nargs="*", default=[], help="assets that must stay in the team")
    ap.add_argument("--avoid", nargs="*", default=[], help="assets never to pick")
    ap.add_argument("--spec", default=FP.SPEC_FILE)
    ap.add_argument("--refresh", action="store_true", help="recompute forecasts (ignore the cache)")
    ap.add_argument("--sharp", action="store_true", help="plan on the raw forecast model, without the planning shrinkage")
    ap.add_argument("--csv", default=None, help="write the plan to this CSV")
    a = ap.parse_args()
    if a.team and len(a.team) != 7:
        ap.error("--team needs exactly 7 names (5 drivers, 2 constructors)")

    import mlflow_experiments as X
    res, truth, _ = X.get_data()
    rounds = FP.upcoming_rounds(res, a.rounds)
    names = [next(r["name"] for r in FP.RACE_CALENDAR_2026 if r["round"] == x) for x in rounds]
    specs, c_sim, spec_name = load_specs(a.spec)
    if a.sharp:
        c_sim = (c_sim[0], c_sim[1], None)
    pts = forecasts(rounds, specs, c_sim, a.refresh)
    overrides = dict(p.split("=", 1) for p in a.price)
    price_now = FP.current_prices(overrides=overrides)
    op = FP.official_tables()
    path = np.repeat(price_now[None, :], len(rounds) + 1, 0) if a.flat_prices else FP.project_prices(price_now, pts, op)
    plan, state = FP.plan(pts, path, team=a.team, bank=a.bank, budget=a.budget, free=a.free, chips=a.chips,
                          discount=a.discount, locked_in=a.keep, locked_out=a.avoid)
    print(f"Model: {spec_name} | races: {', '.join(names)} | {'flat' if a.flat_prices else 'projected'} prices | "
          f"start: {'team ' + ', '.join(a.team) if a.team else f'fresh team, ${a.budget:.1f}m'} | "
          f"value+bank ${state.wealth:.1f}m | {a.free} free transfer(s) | chips {a.chips or 'none'} | optimal: {plan.optimal}")
    rows, prev = [], (set(FP.canon_entity(n) for n in a.team) if a.team else None)
    for k, nm in enumerate(names):
        team = [FP.ENTITIES[e] for e in sorted(plan.teams[k])]
        din = sorted(set(team) - prev) if prev is not None else team
        dout = sorted(prev - set(team)) if prev is not None else []
        cost = float(sum(path[k][FP.ENTITIES.index(x)] for x in team))
        rows.append({"race": nm, "drivers": ", ".join(x for x in team if x in FP.DRIVERS),
                     "constructors": ", ".join(x for x in team if x in FP.TEAMS), "boost": FP.ENTITIES[plan.boosts[k]],
                     "chip": plan.chips[k] or "", "in": ", ".join(din) if prev is not None else "(new team)", "out": ", ".join(dout),
                     "transfers": plan.transfers_in[k], "penalty": plan.penalties[k], "exp_points": round(plan.round_points[k], 1),
                     "team_cost": round(cost, 1)})
        prev = set(team)
    df = pd.DataFrame(rows)
    with pd.option_context("display.width", 250, "display.max_colwidth", 80):
        print(df.to_string(index=False))
    if "autopilot" in a.chips:
        ap_ = FP.autopilot_advice(plan, pts)
        if ap_:
            print(f"Autopilot: best used at {names[ap_[0]]} (+{ap_[1]:.1f} expected over boosting the top-forecast driver)")
    print(f"\nExpected total over {len(rounds)} races: {plan.total:.1f} points (forecast error per driver per race is "
          f"roughly +-11-15 points; re-plan every week with fresh results).")
    nxt = pd.DataFrame({"asset": FP.ENTITIES, "type": np.where(FP.IS_DRIVER, "D", "C"), "exp_points_next": pts[0].round(1),
                        "price": price_now, "pts_per_$m": (pts[0] / price_now).round(2)}).sort_values("exp_points_next", ascending=False)
    print(f"\nForecast for {names[0]}:\n" + nxt.head(16).to_string(index=False))
    if a.csv:
        df.to_csv(a.csv, index=False)


if __name__ == "__main__":
    main()
