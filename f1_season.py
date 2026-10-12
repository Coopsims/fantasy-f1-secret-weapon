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
import pickle
from dataclasses import dataclass, replace

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
    rounds, ts, pts, ev, pit_known, active = [], [], [], [], [], []
    for _, w in wk.iterrows():
        d, c = score_weekend_actual(res, int(w["t"]))
        if d.empty:
            continue
        row = np.zeros(len(ENTITIES))
        e_row = np.zeros((len(ENTITIES), 3))
        d = d.set_index("driver")
        active.append(np.array([n in d.index for n in DRIVERS] + [True] * len(TEAMS)))
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
    return {"rounds": rounds, "t": ts, "pts": np.array(pts), "events": np.array(ev), "active": np.array(active),
            "pit_known": np.array(pit_known)}


def _canon_entity(name):
    from f1_calendar import canon_driver, canon_team
    t = canon_team(name)
    return t if t in TEAMS else canon_driver(name)


def _data_key(res):
    """Fingerprint of the race data, so a cached replay is never reused after the data changes."""
    import hashlib
    cols = [c for c in ("t", "session", "driver", "team", "grid", "finish", "dnf", "fastest_lap", "overtakes") if c in res.sessions]
    h = pd.util.hash_pandas_object(res.sessions[cols], index=False).values
    return hashlib.sha1(h.tobytes()).hexdigest()[:16]


def price_table(rounds, pts, price_csv=None, verbose=False, return_rule=False, active=None):
    """
    price[i] = price before the i-th raced weekend (what you pay when you transfer for it).

    Real prices from the price-history CSV (round k = prices after calendar round k) are
    used wherever they exist; the weekends in between are reconstructed by replaying the
    game's price-change rule (f1_prices), with the rule's unknown cut-offs fitted to the
    real prices.  Cancelled rounds are handled: a label is mapped to the number of raced
    weekends completed by then.  Returns (price matrix, n_estimated).
    """
    from f1_prices import checkpoints_from_csv, reconstruct
    base = np.array([DEFAULT_DRIVER_PRICES[d] for d in DRIVERS] + [DEFAULT_CONSTRUCTOR_PRICES[t] for t in TEAMS])
    anchors = {}
    if price_csv and os.path.exists(price_csv):
        df = pd.read_csv(price_csv)
        df["name"] = df["name"].map(_canon_entity)        # e.g. "Haas" -> "Haas F1 Team"; unmatched names were silently ignored
        anchors = checkpoints_from_csv(df, rounds, ENTITIES)
    P, info = reconstruct(np.asarray(pts, float), base, anchors, verbose=verbose, active=active)
    R = len(rounds)
    n_est = sum(1 for j in range(1, R) if j not in anchors or np.isnan(anchors[j]).any())
    if return_rule:
        return P[:R], n_est, info["rule"]
    return P[:R], n_est


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
    for i, (rnd, t) in enumerate(zip(act["rounds"], act["t"])):
        hist = truncate(res, t)
        assert hist.empty or hist.weekends["t"].max() < t, "future data leaked into the model"
        f = WeekendForecaster(hist, params=params, year=year, verbose=False)
        for k, target in enumerate(act["rounds"][i:i + horizon]):    # next raced weekends (skips cancelled rounds)
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
    # With official points every constructor score includes pit stops, so decisions must use the forecast
    # *with* pit points; reconstructed points only have them where OpenF1 has stop times.
    return {"actuals": act, "forecasts": forecasts, "accuracy": pd.DataFrame(acc_rows),
            "official": official is not None}


# ──────────────────────────────────────────────────────────────
# 2. Play the season with only the information available at the time
# ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PlayPolicy:
    """How the no-hindsight player turns forecasts into decisions (all tunable)."""
    horizon: int = 3            # raced weekends planned ahead
    discount: float = 0.7       # weight of each later weekend in the plan
    price_aware: bool = True    # project prices with the game's rule instead of holding them flat
    value_weight: float = 0.0   # points per $1m of projected team-value growth
    chip_cost: float = 0.0      # reserve value of an unused chip (points)
    ewma_blend: float = 0.0     # shrink the forecast toward an EWMA of actual points
    use_chips: bool = True


def project_prices(price_now, past_ppm, fc_pts, rule):
    """
    Price path over a planning horizon using the game's rule.
    price_now : (E,) actual price before the first planned weekend
    past_ppm  : list of (E,) points-per-$m of the races already run (latest last)
    fc_pts    : (H, E) forecast points of the planned weekends
    Returns (H + 1, E): prices before each planned weekend, then after the last one.
    """
    from f1_prices import step_prices
    P = [np.asarray(price_now, float)]
    ppm = list(past_ppm)
    for k in range(len(fc_pts)):
        ppm.append(np.asarray(fc_pts[k], float) / np.maximum(P[-1], 0.1))
        P.append(step_prices(P[-1], ppm[-rule.window:], rule))
    return np.array(P)


def _ewma(x, alpha=0.2):
    out = x[0].astype(float).copy()
    for row in x[1:]:
        out = alpha * row + (1 - alpha) * out
    return out


def play_season(rep, prices, rules=None, horizon=3, discount=0.7, use_chips=True, verbose=True,
                policy=None, rule=None):
    """
    Rolling-horizon MILP on the forecasts available at the time; real results and real
    prices are revealed one weekend at a time.  `policy` (PlayPolicy) overrides the
    horizon / discount / use_chips arguments.  `rule` (f1_prices.PriceRule) is used for
    price projection when policy.price_aware.
    """
    from f1_prices import PriceRule
    rules = rules or Rules()
    policy = policy or PlayPolicy(horizon=horizon, discount=discount, use_chips=use_chips, price_aware=False)
    rule = rule or PriceRule()
    act = rep["actuals"]
    R = len(act["rounds"])
    state = State(prev_team=None, wealth=rules.budget, free_transfers=rules.free_per_round,
                  chips_left=rules.chips if policy.use_chips else ())
    chip_cost = ({c: policy.chip_cost for c in rules.chips} if policy.chip_cost else None)
    log, total = [], 0.0
    for i, rnd in enumerate(act["rounds"]):
        H = [k for k in range(policy.horizon)
             if i + k < R and (rnd, act["rounds"][i + k]) in rep["forecasts"]]
        fcs = [rep["forecasts"][(rnd, act["rounds"][i + k])] for k in H]
        pts_h = np.array([f["mean"] + (f["pit"] if rep.get("official") else 0.0) for f in fcs])
        if policy.ewma_blend and i > 0:
            pts_h = (1 - policy.ewma_blend) * pts_h + policy.ewma_blend * _ewma(act["pts"][:i])[None, :]
        if policy.price_aware:
            past = [act["pts"][j] / np.maximum(prices[j], 0.1) for j in range(max(0, i - rule.window + 1), i)]
            path = project_prices(prices[i], past, pts_h, rule)
            price_h, terminal = path[:-1], path[-1]
        else:
            price_h, terminal = np.array([prices[i]] * len(H)), prices[i]
        disc = policy.discount ** np.arange(len(H))
        plan = solve_strategy(pts_h, price_h, IS_DRIVER, rules, state, disc, time_limit=60, mip_rel_gap=1e-6,
                              terminal_price=terminal, value_weight=policy.value_weight, chip_cost=chip_cost)
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


def _play_one(args):
    rep, prices, rules, policy, rule = args
    out = play_season(rep, prices, rules, verbose=False, policy=policy, rule=rule)
    return out["total"], list(out["log"]["points"])


def default_policy_grid():
    from itertools import product
    grid = []
    for h, d, va, vw, cc, ew in product((1, 2, 3, 4), (0.6, 0.85), (False, True), (0.0, 5.0, 15.0),
                                        (0.0, 15.0), (0.0, 0.3)):
        if not va and vw:
            continue                      # value weight needs projected prices
        grid.append(PlayPolicy(horizon=h, discount=d, price_aware=va, value_weight=vw, chip_cost=cc, ewma_blend=ew))
    return grid


def tune_play(rep, prices, rules=None, rule=None, grid=None, workers=None, optimum=None, verbose=True):
    """
    Play the season once per policy and measure the gap to the hindsight optimum.

    Selecting the policy on the same weekends it is scored on flatters it, so two honest
    columns are added: the gap on the first and second half of the season separately
    (`sel_first_half` picks on the first half and reports how that pick did on the second).
    """
    from concurrent.futures import ProcessPoolExecutor
    rules = rules or Rules()
    grid = grid or default_policy_grid()
    act = rep["actuals"]
    if optimum is None:
        optimum, _ = hindsight_optimum(act, prices, rules)
    opt_round = np.array(optimum.round_points)
    half = len(opt_round) // 2
    jobs = [(rep, prices, rules, p, rule) for p in grid]
    if workers == 1:
        results = [_play_one(j) for j in jobs]
    else:
        # "spawn", not fork: the optimum solve above leaves HiGHS threads behind, and forked
        # children inherit their locks and can hang forever.
        import multiprocessing as mp
        with ProcessPoolExecutor(max_workers=workers or os.cpu_count(), mp_context=mp.get_context("spawn")) as ex:
            results = list(ex.map(_play_one, jobs, chunksize=4))
    rows = []
    for gi, (p, (tot, per)) in enumerate(zip(grid, results)):
        per = np.array(per)
        rows.append({**p.__dict__, "grid_index": gi, "points": tot, "delta": optimum.total - tot,
                     "delta_first_half": opt_round[:half].sum() - per[:half].sum(),
                     "delta_second_half": opt_round[half:].sum() - per[half:].sum()})
    df = pd.DataFrame(rows).sort_values("delta").reset_index(drop=True)
    if verbose:
        print(f"\n  Hindsight optimum {optimum.total:.0f}.  {len(df)} policies played.  Best 10 by gap:")
        cols = ["horizon", "discount", "price_aware", "value_weight", "chip_cost", "ewma_blend", "points", "delta",
                "delta_first_half", "delta_second_half"]
        print(df[cols].head(10).to_string(index=False))
        pick = df.sort_values("delta_first_half").iloc[0]
        print(f"\n  Honest check: best policy on the first {half} weekends -> gap on the remaining "
              f"{len(opt_round) - half}: {pick['delta_second_half']:.0f} "
              f"(the in-sample best scores {df.iloc[0]['delta_second_half']:.0f} there).")
    return df


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

def _replay_key(res, year, n_sims, horizon, params, official):
    off = None if official is None else int(pd.util.hash_pandas_object(official, index=False).sum())
    return (year, n_sims, horizon, _data_key(res), repr(params), off)


def cached_replay(res, year, n_sims, horizon, params=None, official=None, cache=None):
    """replay(), stored in / loaded from `cache` (pickle) so policy tuning does not refit the model."""
    if cache and os.path.exists(cache):
        with open(cache, "rb") as f:
            rep = pickle.load(f)
        if rep.get("_key") == _replay_key(res, year, n_sims, horizon, params, official):
            rep["official"] = official is not None
            print(f"  (replay loaded from {cache})")
            return rep
    rep = replay(res, year, n_sims, horizon, params, official=official)
    rep["_key"] = _replay_key(res, year, n_sims, horizon, params, official)
    if cache:
        with open(cache, "wb") as f:
            pickle.dump(rep, f)
    return rep


def run_all(res, year, prices_csv=None, official_csv=None, n_sims=4000, horizon=3,
            rules=None, out_prefix="season", params=None, policy=None, cache=None):
    official = pd.read_csv(official_csv) if official_csv else None
    print("\n══ 1. Walk-forward replay (each weekend rebuilt from earlier weekends only) ══")
    rep = cached_replay(res, year, n_sims, max(horizon, policy.horizon if policy else 0), params, official, cache)
    acc = rep["accuracy"]
    if acc.empty:
        raise SystemExit(f"No completed {year} weekends in the data.")
    act = rep["actuals"]
    prices, n_est, rule = price_table(act["rounds"], act["pts"], prices_csv, verbose=True, return_rule=True, active=act.get("active"))
    print(f"\n  Driver MAE {acc.driver_mae.mean():.2f} vs naive-last-3 {acc.driver_mae_naive3.mean():.2f}; "
          f"rank-corr {acc.driver_rankcorr.mean():.2f}; 80% cover {acc.driver_cover80.mean():.2f}; "
          f"constructor MAE {acc.constructor_mae.mean():.2f} vs {acc.constructor_mae_naive3.mean():.2f}")
    if n_est:
        print(f"  ⚠ {n_est} weekend(s) of prices were ESTIMATED (tier model); supply {prices_csv or 'a price CSV'} for exact budgets.")
    print("\n══ 2. Playing the season with only the information available at the time ══")
    play = play_season(rep, prices, rules, horizon, policy=policy, rule=rule)
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
    ap.add_argument("cmd", choices=["all", "tune"],
                    help="all = replay + play + optimum; tune = search play policies for the smallest gap to the optimum")
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--prices", default="fantasy_f1_prices.csv")
    ap.add_argument("--official-points", help="CSV(round,name,points) of the game's real points")
    ap.add_argument("--sims", type=int, default=4000)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--free-per-round", type=int, default=2)
    ap.add_argument("--max-bank", type=int, default=3)
    ap.add_argument("--workers", type=int, default=None, help="parallel processes for tune")
    ap.add_argument("--cache", default=None, help="pickle file to store/reuse the replay (default season_<year>_replay.pkl)")
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
    cache = a.cache or (None if a.demo else f"season_{a.year}_replay.pkl")
    if a.cmd == "all":
        run_all(res, a.year, a.prices, a.official_points, a.sims, a.horizon, rules, a.out, cache=cache)
        return
    official = pd.read_csv(a.official_points) if a.official_points else None
    grid = default_policy_grid()
    rep = cached_replay(res, a.year, a.sims, max(p.horizon for p in grid), official=official, cache=cache)
    act = rep["actuals"]
    prices, n_est, rule = price_table(act["rounds"], act["pts"], a.prices, verbose=True, return_rule=True, active=act.get("active"))
    if n_est:
        print(f"  ⚠ {n_est} weekend(s) of prices reconstructed by the rule (not real prices).")
    df = tune_play(rep, prices, rules, rule, grid, workers=a.workers)
    df.to_csv(f"{a.out}_policy_search.csv", index=False)
    print(f"\n  Wrote {a.out}_policy_search.csv")


if __name__ == "__main__":
    _cli()
