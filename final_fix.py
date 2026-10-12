#!/usr/bin/env python3
"""
Final Fix advisor (final_fix.py)
================================
The Final Fix chip allows one change between the end of Qualifying and the start of the Grand Prix, with no transfer
penalty (game FAQ).  This ranks every affordable swap by the points still to come: the weekend simulator is
conditioned on the REAL qualifying order (from the OpenF1 cache once Qualifying has run, or --quali) and only the
Grand Prix is scored (incl. Driver of the Day; constructors: their race points incl. pit stops).  The asset taken out
keeps what it already scored; if the boosted driver goes, the boost passes to the incoming driver (game FAQ).

    python fetch_openf1_full.py                      # after Qualifying
    python final_fix.py --team Antonelli Bearman Hulkenberg Lawson Lindblad Ferrari Mercedes --boost Antonelli --bank 0.3

Drivers only by default (how the chip is described in the guides); --constructors also tries constructor swaps.
--not-starting NAME: a driver who will not take the start scores the not-classified -20 (game rules: the penalty
applies to drivers not on the final starting grid), and so does his constructor's share.

When it matters: on average the real grid hardly sharpens the race forecast (2026 check, experiments/final_fix_check.py:
RMSE 13.18 -> 13.15, CI -0.21..+0.13), so the chip pays mostly when something happens to one of your assets
(a crash in qualifying, a grid penalty, a driver not starting).
"""
import argparse

import numpy as np
import pandas as pd

import f1_planner as FP


def qualifying_order(rnd, full=None):
    """Real qualifying order (best first) for 2026 round `rnd` from the OpenF1 cache, or None if not there yet."""
    import openf1_cache as C
    from f1_calendar import canon_driver, canon_location
    from f1_weekend_data import _latest_positions, _relabel_sessions
    race = next(r for r in FP.RACE_CALENDAR_2026 if r["round"] == rnd)
    full = full if full is not None else C.load_all(only_complete=False)
    for m in C.to_legacy_meetings(full):
        if m["year"] != 2026 or canon_location(m["location"]) != canon_location(race["location"]):
            continue
        for code, s in _relabel_sessions(m["sessions"]):
            if code != "Q":
                continue
            dmap = {d["driver_number"]: canon_driver(d.get("last_name", "")) for d in (s.get("drivers") or m["drivers"])}
            latest = _latest_positions(s.get("positions"))
            order = [dmap[dn] for dn, _ in sorted(latest.items(), key=lambda kv: kv[1]) if dmap.get(dn)]
            return order or None
    return None


def final_fix_options(team, boost, bank, price, race_draws, allow_constructors=False):
    """Every affordable single swap ranked by expected Grand Prix gain.
    race_draws: {asset: array of simulated GP points (same draws for every asset)}.  Returns a DataFrame."""
    team = list(team)
    rows = []
    for out in team:
        if out not in FP.DRIVERS and not allow_constructors:
            continue
        pool = FP.DRIVERS if out in FP.DRIVERS else FP.TEAMS
        mult = 2.0 if out == boost else 1.0
        for inn in pool:
            if inn in team or inn not in race_draws or out not in race_draws:
                continue
            p_in, p_out = price[FP.ENTITIES.index(inn)], price[FP.ENTITIES.index(out)]
            if p_in > p_out + bank + 1e-9:
                continue
            d = mult * (np.asarray(race_draws[inn]) - np.asarray(race_draws[out]))
            rows.append({"out": out, "in": inn, "boosted": out == boost, "exp_gain": float(d.mean()),
                         "p_better": float((d > 0).mean()), "price_in": p_in, "price_out": p_out})
    df = pd.DataFrame(rows)
    return df.sort_values("exp_gain", ascending=False).reset_index(drop=True) if len(df) else df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--team", nargs=7, required=True)
    ap.add_argument("--boost", required=True, help="your 2x driver")
    ap.add_argument("--bank", type=float, required=True, help="cost cap remaining ($m)")
    ap.add_argument("--round", type=int, default=None, help="2026 round (default: the next race)")
    ap.add_argument("--quali", default=None, help="comma-separated qualifying order, best first (default: from OpenF1)")
    ap.add_argument("--constructors", action="store_true", help="also try constructor swaps")
    ap.add_argument("--price", action="append", default=[], help="override a price, e.g. --price Antonelli=26.5")
    ap.add_argument("--not-starting", nargs="*", default=[], help="drivers who will not start the Grand Prix")
    ap.add_argument("--penalty", action="append", default=[], help="grid penalty, e.g. --penalty Norris=5")
    ap.add_argument("--sims", type=int, default=20000)
    a = ap.parse_args()

    import mlflow_experiments as X
    from f1_calendar import canon_driver
    from f1_weekend_model import WeekendForecaster
    res, _, _ = X.get_data()
    rnd = a.round or FP.upcoming_rounds(res, 1)[0]
    name = next(r["name"] for r in FP.RACE_CALENDAR_2026 if r["round"] == rnd)
    order = [canon_driver(x.strip()) for x in a.quali.split(",")] if a.quali else qualifying_order(rnd)
    if not order:
        raise SystemExit(f"No qualifying result for {name} yet: run fetch_openf1_full.py after Qualifying, or pass --quali")
    pen = {canon_driver(k): int(v) for k, v in (x.split("=", 1) for x in a.penalty)}
    fc = WeekendForecaster(res, year=2026, verbose=False).forecast(rnd, n=a.sims, seed=500 + rnd, known={"quali": order},
                                                                   grid_penalties=pen or None)
    race = fc.event_points("race", include_dotd=True)
    draws = {d: race[:, i] for i, d in enumerate(fc.drivers)}
    craces = fc.constructor_event_points("race")
    draws |= {t: craces[:, j].astype(float) for j, t in enumerate(fc.teams)}
    from f1_calendar import GRID_2026
    for d in (canon_driver(x) for x in a.not_starting):
        if d in draws:
            if GRID_2026.get(d) in draws:          # the constructor scores its drivers' total: swap in the -20
                draws[GRID_2026[d]] = draws[GRID_2026[d]] - fc.event_points("race", include_dotd=False)[:, fc.drivers.index(d)] - 20.0
            draws[d] = np.full(a.sims, -20.0)
    team = [FP.canon_entity(n) for n in a.team]
    boost = FP.canon_entity(a.boost)
    price = FP.current_prices(overrides=dict(p.split("=", 1) for p in a.price))
    opts = final_fix_options(team, boost, a.bank, price, draws, a.constructors)
    print(f"{name}: qualifying order used: {', '.join(order[:10])}{' ...' if len(order) > 10 else ''}")
    print("Your team's expected Grand Prix points from here: "
          + ", ".join(f"{t} {np.mean(draws[t]) * (2 if t == boost else 1):.1f}" for t in team if t in draws))
    if not len(opts):
        print("No affordable swap.")
        return
    b = opts.iloc[0]
    if b.exp_gain <= 0:
        print("\nNo swap is expected to gain anything - keep the Final Fix chip.")
    else:
        print(f"\nBest Final Fix: OUT {b.out} IN {b['in']}{' (takes the 2x boost)' if b.boosted else ''} -> +{b.exp_gain:.1f} "
              f"expected points; better in {b.p_better:.0%} of simulated races. Worth playing if that beats what you expect "
              f"to save it for (e.g. one of your drivers starting from the back).")
    print("\nTop swaps:\n" + opts.head(10).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
