#!/usr/bin/env python3
"""
Season replay + strategy (f1_season.py)
=======================================
1. REPLAY    Walk the season weekend by weekend.  Before each weekend the model is
             rebuilt and recalibrated from scratch using ONLY earlier weekends
             (a leakage guard asserts this), then forecasts that weekend (and the
             next few).  Forecasts are scored against what really happened.
2. PLAY      Plays the season as the system would have: each weekend it plans with
             the forecasts available *at that time* (rolling horizon, exact MILP),
             commits that weekend's team / boost / chip / transfers, then sees the
             real result, the real prices and moves on.
3. OPTIMUM   With hindsight, the provably optimal season (f1_strategy.solve_strategy).

    python f1_season.py all --year 2026 [--prices fantasy_f1_prices.csv]
    python f1_season.py all --demo            # synthetic data, no files needed

Points come from the OpenF1 data via f1_weekend_data (Driver of the Day and - unless
stop times are present - pit-stop points are not observable, so forecasts are made on
the same basis).  Pass --official-points to use a CSV of the game's real points.
"""

import argparse
import os
from dataclasses import replace

import numpy as np
import pandas as pd

from f1_calendar import (GRID_2026, DEFAULT_DRIVER_PRICES, DEFAULT_CONSTRUCTOR_PRICES,
                         RACE_CALENDAR_2026)
from f1_strategy import Rules, State, Plan, solve_strategy, evaluate_plan
from f1_weekend_data import Results, load_results, score_weekend_actual
from f1_weekend_model import ModelParams, WeekendForecaster

DRIVERS = list(GRID_2026)
TEAMS = sorted(set(GRID_2026.values()))
ENTITIES = DRIVERS + TEAMS
IS_DRIVER = np.array([True] * len(DRIVERS) + [False] * len(TEAMS))


# ──────────────────────────────────────────────────────────────
# Actual points and prices
# ──────────────────────────────────────────────────────────────

def truncate(res, before_t):
    """Results containing only weekends strictly before weekend index `before_t`."""
    wk = res.weekends[res.weekends["t"] < before_t]
    s = res.sessions[res.sessions["t"] < before_t]
    p = res.pits[res.pits["t"] < before_t] if res.pits is not None and not res.pits.empty else res.pits
    return Results(s, p, wk)


def season_actuals(res, year, official=None):
    """
    {rounds, t, pts (R x E), events {name: (R x 3)}, pit_known (R)} for completed weekends.
    `official` = optional DataFrame(round, name, points) overriding reconstructed totals.
    """
    wk = res.weekends[res.weekends["year"] == year].sort_values("round")
    rounds, ts, pts, ev, pit_known = [], [], [], [], []
    for _, w in wk.iterrows():
        d, c = score_weekend_actual(res, int(w["t"]))
        if d.empty:
            continue
        row = np.zeros(len(ENTITIES))
        e_row = np.zeros((len(ENTITIES), 3))
        d = d.set_index("driver")
        for i, name in enumerate(DRIVERS):
            if name in d.index:
                row[i] = d.loc[name, "total"]
                e_row[i] = d.loc[name, ["quali", "sprint", "race"]].values
        c = c.set_index("team")
        pk = bool(c["pit_known"].all()) if len(c) else False
        for j, name in enumerate(TEAMS):
            if name in c.index:
                row[len(DRIVERS) + j] = c.loc[name, "total"]
                e_row[len(DRIVERS) + j] = c.loc[name, ["quali", "sprint", "race"]].values
        if official is not None:
            o = official[official["round"] == w["round"]].set_index("name")["points"]
            for i, name in enumerate(ENTITIES):
                if name in o.index:
                    row[i] = o[name]
        rounds.append(int(w["round"])); ts.append(int(w["t"]))
        pts.append(row); ev.append(e_row); pit_known.append(pk)
    return {"rounds": rounds, "t": ts, "pts": np.array(pts), "events": np.array(ev),
            "pit_known": np.array(pit_known)}


def _tier_change(rank_pct, price):
    tier = 0.3 if price >= 20.0 else 0.6
    lvl = 1.0 if rank_pct >= 0.75 else (1 / 3 if rank_pct >= 0.5 else (-1 / 3 if rank_pct >= 0.25 else -1.0))
    return round(tier * lvl, 1)


def estimate_next_prices(prices, pts_row):
    """Approximate the game's tier-based price move from one weekend's points (per type)."""
    out = prices.copy()
    for mask in (IS_DRIVER, ~IS_DRIVER):
        idx = np.flatnonzero(mask)
        vals = pts_row[idx]
        for i in idx:
            pct = (vals <= pts_row[i]).mean()
            out[i] = max(4.0, round(prices[i] + _tier_change(pct, prices[i]), 1))
    return out


def price_table(rounds, pts, price_csv=None):
    """
    price[i] = price before the i-th weekend (what you pay when you transfer for it).
    Uses the price-history CSV (round k = prices after round k) where present and
    estimates the rest from actual points.  Returns (price matrix, n_estimated).
    """
    base = np.array([DEFAULT_DRIVER_PRICES[d] for d in DRIVERS] + [DEFAULT_CONSTRUCTOR_PRICES[t] for t in TEAMS])
    table = {}
    if price_csv and os.path.exists(price_csv):
        df = pd.read_csv(price_csv)
        for k, g in df.groupby("round"):
            table[int(k)] = np.array([g.set_index("name")["price"].get(n, np.nan) for n in ENTITIES])
    prices, est = [], 0
    cur = table.get(0, base).copy()
    cur = np.where(np.isnan(cur), base, cur)
    for i, rnd in enumerate(rounds):
        k = rnd - 1
        if k in table:
            cur = np.where(np.isnan(table[k]), cur, table[k])
        elif i > 0:
            cur = estimate_next_prices(prices[-1], pts[i - 1])
            est += 1
        prices.append(cur.copy())
    return np.array(prices), est


# ──────────────────────────────────────────────────────────────
# 1. Walk-forward replay
# ──────────────────────────────────────────────────────────────

def replay(res, year=2026, n_sims=4000, horizon=3, params=None, verbose=True, official=None):
    """
    Returns dict: actuals, forecasts {(as_of_round, target_round): dict of arrays}, accuracy DataFrame.
    A fresh WeekendForecaster (=> fresh ratings AND fresh calibration) is built per weekend
    from earlier weekends only.
    """
    params = replace(params or ModelParams(), dotd=official is not None)
    act = season_actuals(res, year, official)
    forecasts, acc_rows = {}, []
    n_cal = len(RACE_CALENDAR_2026)
    for i, (rnd, t) in enumerate(zip(act["rounds"], act["t"])):
        hist = truncate(res, t)
        assert hist.empty or hist.weekends["t"].max() < t, "future data leaked into the model"
        f = WeekendForecaster(hist, params=params, year=year, verbose=False)
        for k in range(horizon):
            target = rnd + k
            if target > n_cal:
                break
            fc = f.forecast(target, n=n_sims, seed=100 + target)
            d_ev = np.stack([fc.event_points(e, include_dotd=params.dotd).mean(0) for e in ("quali", "sprint", "race")], 1)
            tot = fc.total_points(include_dotd=params.dotd)
            c_pit = fc.ccomp[("race", "pit")]
            c_tot = fc.constructor_total()
            order_d = [fc.drivers.index(d) for d in DRIVERS]
            order_c = [fc.teams.index(c) for c in TEAMS]
            forecasts[(rnd, target)] = {
                "mean": np.r_[tot.mean(0)[order_d], (c_tot - c_pit).mean(0)[order_c]],
                "pit": np.r_[np.zeros(len(DRIVERS)), c_pit.mean(0)[order_c]],
                "sd": np.r_[tot.std(0)[order_d], (c_tot - c_pit).std(0)[order_c]],
                "p10": np.r_[np.percentile(tot, 10, 0)[order_d], np.percentile(c_tot - c_pit, 10, 0)[order_c]],
                "p90": np.r_[np.percentile(tot, 90, 0)[order_d], np.percentile(c_tot - c_pit, 90, 0)[order_c]],
                "events_d": d_ev[order_d],
            }
            if k == 0:
                fc0 = forecasts[(rnd, target)]
                a = act["pts"][i]
                adj = fc0["mean"] + (fc0["pit"] if act["pit_known"][i] or official is not None else 0)
                lo = fc0["p10"] + (fc0["pit"] if act["pit_known"][i] or official is not None else 0)
                hi = fc0["p90"] + (fc0["pit"] if act["pit_known"][i] or official is not None else 0)
                row = {"round": rnd, "history_weekends": int(len(hist.weekends))}
                for nm, mask in (("driver", IS_DRIVER), ("constructor", ~IS_DRIVER)):
                    p, aa = adj[mask], a[mask]
                    prev = act["pts"][max(0, i - 3):i][:, mask].mean(0) if i > 0 else np.full(mask.sum(), aa.mean())
                    row[f"{nm}_mae"] = float(np.abs(p - aa).mean())
                    row[f"{nm}_mae_naive3"] = float(np.abs(prev - aa).mean())
                    row[f"{nm}_rankcorr"] = float(pd.Series(p).rank().corr(pd.Series(aa).rank()))
                    row[f"{nm}_cover80"] = float(((lo[mask] <= aa) & (aa <= hi[mask])).mean())
                ev = act["events"][i][IS_DRIVER]
                for j, nm in enumerate(("quali", "sprint", "race")):
                    row[f"driver_mae_{nm}"] = float(np.abs(fc0["events_d"][:, j] - ev[:, j]).mean())
                acc_rows.append(row)
                if verbose:
                    print(f"  R{rnd:<2d} trained on {row['history_weekends']:>2d} weekends | driver MAE {row['driver_mae']:5.2f} "
                          f"(naive {row['driver_mae_naive3']:5.2f}) | rank-corr {row['driver_rankcorr']:5.2f} "
                          f"| 80%-cover {row['driver_cover80']:.2f}")
    return {"actuals": act, "forecasts": forecasts, "accuracy": pd.DataFrame(acc_rows)}


# ──────────────────────────────────────────────────────────────
# 2. Play the season with only the information available at the time
# ──────────────────────────────────────────────────────────────

def play_season(rep, prices, rules=None, horizon=3, discount=0.7, use_chips=True, verbose=True):
    """Rolling-horizon MILP on forecasts; realised results/prices revealed one weekend at a time."""
    rules = rules or Rules()
    act = rep["actuals"]
    R = len(act["rounds"])
    state = State(prev_team=None, wealth=rules.budget, free_transfers=rules.free_per_round,
                  chips_left=rules.chips if use_chips else ())
    log, total = [], 0.0
    for i, rnd in enumerate(act["rounds"]):
        H = [k for k in range(horizon) if (rnd, rnd + k) in rep["forecasts"] and i + k < R]
        pts_h = np.array([rep["forecasts"][(rnd, rnd + k)]["mean"] for k in H])
        price_h = np.array([prices[i]] * len(H))                      # future prices unknown: hold flat
        disc = discount ** np.arange(len(H))
        plan = solve_strategy(pts_h, price_h, IS_DRIVER, rules, state, disc, time_limit=60, mip_rel_gap=1e-6)
        team, boost, chip = plan.teams[0], plan.boosts[0], plan.chips[0]
        # realise weekend i
        r, per, pen, trs, ok = evaluate_plan([team], [boost], [chip], act["pts"][i][None, :],
                                             prices[i][None, :], IS_DRIVER, rules, state)
        assert ok
        total += r
        log.append({"round": rnd, "team": sorted(team), "boost": boost, "chip": chip,
                    "transfers": trs[0], "penalty": pen[0], "points": per[0], "wealth": state.wealth or rules.budget})
        # roll the state forward with the real prices
        W = (state.wealth if state.wealth is not None else rules.budget)
        nxt = prices[i + 1] if i + 1 < R else prices[i]
        used = 0 if (state.prev_team is None or chip == "wildcard") else min(trs[0], state.free_transfers)
        state = State(prev_team=team,
                      wealth=W + sum(nxt[e] - prices[i][e] for e in team),
                      free_transfers=min(rules.max_bank, state.free_transfers - used + rules.free_per_round),
                      chips_left=tuple(c for c in state.chips_left if c != chip))
        if verbose:
            names = [ENTITIES[e] for e in sorted(team)]
            print(f"  R{rnd:<2d} {per[0]:6.1f} pts  boost {ENTITIES[boost]:<11s} chip {chip or '-':<12s} "
                  f"transfers {trs[0]} (-{pen[0]:.0f})  {', '.join(names)}")
    return {"log": pd.DataFrame(log), "total": total}


# ──────────────────────────────────────────────────────────────
# 3. Hindsight optimum
# ──────────────────────────────────────────────────────────────

def hindsight_optimum(act, prices, rules=None, time_limit=600):
    rules = rules or Rules()
    plan = solve_strategy(act["pts"], prices, IS_DRIVER, rules, time_limit=time_limit)
    rows = []
    for i, rnd in enumerate(act["rounds"]):
        rows.append({"round": rnd, "team": [ENTITIES[e] for e in sorted(plan.teams[i])],
                     "boost": ENTITIES[plan.boosts[i]], "chip": plan.chips[i],
                     "transfers": plan.transfers_in[i], "penalty": plan.penalties[i],
                     "points": plan.round_points[i]})
    return plan, pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def run_all(res, year, prices_csv=None, official_csv=None, n_sims=4000, horizon=3,
            rules=None, out_prefix="season", params=None):
    official = pd.read_csv(official_csv) if official_csv else None
    print("\n══ 1. Walk-forward replay (each weekend rebuilt from earlier weekends only) ══")
    rep = replay(res, year, n_sims, horizon, params, official=official)
    acc = rep["accuracy"]
    if acc.empty:
        raise SystemExit(f"No completed {year} weekends in the data.")
    act = rep["actuals"]
    prices, n_est = price_table(act["rounds"], act["pts"], prices_csv)
    print(f"\n  Driver MAE {acc.driver_mae.mean():.2f} vs naive-last-3 {acc.driver_mae_naive3.mean():.2f}; "
          f"rank-corr {acc.driver_rankcorr.mean():.2f}; 80% cover {acc.driver_cover80.mean():.2f}; "
          f"constructor MAE {acc.constructor_mae.mean():.2f} vs {acc.constructor_mae_naive3.mean():.2f}")
    if n_est:
        print(f"  ⚠ {n_est} weekend(s) of prices were ESTIMATED (tier model); supply {prices_csv or 'a price CSV'} for exact budgets.")
    print("\n══ 2. Playing the season with only the information available at the time ══")
    play = play_season(rep, prices, rules, horizon)
    print("\n══ 3. Hindsight-optimal season (exact, branch-and-bound) ══")
    plan, opt = hindsight_optimum(act, prices, rules)
    for _, r in opt.iterrows():
        print(f"  R{r['round']:<2d} {r['points']:6.1f} pts  boost {r['boost']:<11s} chip {r['chip'] if isinstance(r['chip'], str) else '-':<12s} "
              f"transfers {r['transfers']} (-{r['penalty']:.0f})  {', '.join(r['team'])}")
    print(f"\n  Optimality proven: {plan.optimal} ({plan.status.strip()})")
    print(f"\n  ── Season totals ({len(act['rounds'])} weekends) ──")
    print(f"  Hindsight optimum        : {plan.total:8.1f}")
    print(f"  Model, no future knowledge: {play['total']:8.1f}   ({play['total'] / plan.total:.0%} of optimum)")
    if os.path.exists("fantasy_f1_history.csv"):
        h = pd.read_csv("fantasy_f1_history.csv")
        print(f"  Your recorded team       : {h['actual_points'].sum():8.1f}")
    acc.to_csv(f"{out_prefix}_accuracy.csv", index=False)
    play["log"].to_csv(f"{out_prefix}_model_play.csv", index=False)
    opt.to_csv(f"{out_prefix}_optimum.csv", index=False)
    print(f"\n  Wrote {out_prefix}_accuracy.csv, {out_prefix}_model_play.csv, {out_prefix}_optimum.csv")
    return rep, play, plan


def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["all"])
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--prices", default="fantasy_f1_prices.csv")
    ap.add_argument("--official-points", help="CSV(round,name,points) of the game's real points")
    ap.add_argument("--sims", type=int, default=4000)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--free-per-round", type=int, default=2)
    ap.add_argument("--max-bank", type=int, default=3)
    ap.add_argument("--demo", action="store_true", help="synthetic data (no files needed)")
    ap.add_argument("--out", default="season")
    a = ap.parse_args()
    if a.demo:
        from f1_synthetic import synthetic_results
        res = synthetic_results(seed=7, rounds_2026=8)
        print("  (SYNTHETIC data - not the real season)")
    else:
        res = load_results()
        if res.empty:
            raise SystemExit("No race data found. Need raw_historical_data.json + raw_2026_results.json "
                             "(run f1_trainer.py / the optimizer with OpenF1 access) or the training CSV.")
    rules = Rules(free_per_round=a.free_per_round, max_bank=a.max_bank)
    run_all(res, a.year, a.prices, a.official_points, a.sims, a.horizon, rules, a.out)


if __name__ == "__main__":
    _cli()
