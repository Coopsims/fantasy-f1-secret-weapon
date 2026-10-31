#!/usr/bin/env python3
"""
How the model would have played 2026, with every chip it can play (season_model_test.py)
========================================================================================
The season played walk-forward the way the planner plays it: at each gameday it knows only earlier results and that
weekend's practice, forecasts the next 3 races, solves exactly, plays one week and rolls on.  Scored with the game's
official points and prices, from $100m at the first race.  The earlier replay (4143) played Wildcard, x3 Boost and
No Negative only; this one adds

  Limitless   tried in every week of the 3-race window, as plan_next_race.py does: that week the forecast's best
              team regardless of budget; the real team is parked and comes back, no free transfers carry over
  Autopilot   played in a week without another chip when the planner rates it best in the window; the 2x then goes
              to the team's top-scoring driver

Final Fix is not played (it needs a forecast conditioned on the real qualifying order of every weekend).

Forecasts: the walk-forward ones of experiments/s26_season_replay.py (cached as results/cache/
season_replay_forecasts_*.pkl - each made with only the data before that gameday); planned on the planning blend
(0.5 model + 0.5 x (simulator + 0.4 recent form)), as deployed.

    python season_model_test.py --top 4994
    python season_model_test.py --top 4994 --forecasts ../fantasy-f1-pr/results/cache/season_replay_forecasts_2026-10-12.pkl

No cached forecasts: `python experiments/s26_season_replay.py results/s26_final_spec.json` makes them.
The ceiling (the best possible season, Limitless included) comes from season_optimum.py.
"""
import argparse
import glob
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import f1_planner as FP
from f1_calendar import RACE_CALENDAR_2026
from f1_strategy import Rules, State, evaluate_plan, solve_strategy
from season_optimum import OUT_CSV as OPTIMUM_CSV, load_official

SPRINT = {r["round"]: bool(r.get("sprint")) for r in RACE_CALENDAR_2026}
OUT_CSV = os.path.join("results", "season_2026_model_test.csv")
LOG_CSV = os.path.join("results", "season_2026_model_test_play.csv")

# name -> how Limitless and Autopilot are played (Wildcard, x3 Boost and No Negative always: the solver times them)
POLICIES = {
    "Wildcard, x3, No Negative (the earlier replay)": dict(limitless=None, autopilot=False),
    "+ Limitless": dict(limitless="window", autopilot=False),
    "+ Limitless + Autopilot (the planner as deployed)": dict(limitless="window", autopilot=True),
    "+ Autopilot, Limitless on a Sprint weekend only": dict(limitless="sprint", autopilot=True),
    "+ Autopilot, Limitless held for a 40-point edge": dict(limitless="window", reserve=40.0, autopilot=True),
    "+ Autopilot, Limitless held for an 80-point edge": dict(limitless="window", reserve=80.0, autopilot=True),
}
HEADLINE = "+ Limitless + Autopilot (the planner as deployed)"


def play(fc, rounds, P, C, limitless=None, reserve=0.0, autopilot=False, discount=0.6, time_limit=60, gap=1e-6):
    """Walk-forward season.  fc[t]: forecasts (H x E) made before gameday t for it and the races after; rounds[t]: their
    calendar rounds.  P, C: official points and prices (G x E).  limitless: None (not played), "window" (best week of
    the window) or "sprint" (Sprint weekends only); reserve: forecast edge Limitless must beat.  Returns (total, log)."""
    rules = Rules()                                  # Wildcard, x3 Boost, No Negative: inside the solver
    G = len(P)
    state = State(prev_team=None, wealth=rules.budget, free_transfers=rules.free_per_round, chips_left=rules.chips)
    lim_left, ap_left = limitless is not None, autopilot
    total, log = 0.0, []

    def solve(pts, path, disc, frozen=()):
        return solve_strategy(pts, path, FP.IS_DRIVER, rules, state, disc, time_limit=time_limit, mip_rel_gap=gap,
                              frozen=frozen)

    for t in range(G):
        Hh = min(len(fc[t]), G - t)
        pts = np.array(fc[t][:Hh], float)
        path = np.array([C[t]] * Hh)                 # future prices unknown: held flat
        disc = discount ** np.arange(Hh)
        best, k_lim, lim = solve(pts, path, disc), None, None
        value = best.total
        if lim_left:
            for k in range(Hh):
                if k == 0 and state.prev_team is None:          # nothing to park before the first pick
                    continue
                if limitless == "sprint" and not SPRINT.get(int(rounds[t][k]), False):
                    continue
                lt, lb, lv = FP.limitless_team(pts[k])
                pk = pts.copy()
                pk[k] = 0.0                                     # the parked real team does not score that week
                cand = solve(pk, path, disc, frozen=(k,))
                v = cand.total + disc[k] * (lv - reserve)
                if np.isfinite(cand.total) and v > value + 1e-9:
                    best, k_lim, lim, value = cand, k, (lt, lb, lv), v
        W = state.wealth
        moves = (C[t + 1] - C[t]) if t + 1 < G else np.zeros(len(FP.ENTITIES))
        if k_lim == 0:                               # Limitless this week
            lt, lb, _ = lim
            team = state.prev_team                   # parked: it comes back next week
            r = float(P[t][list(lt)].sum() + P[t][lb])
            row = {"chip": "limitless", "transfers": 0, "penalty": 0.0, "team": lt, "boost": FP.ENTITIES[lb]}
            lim_left = False
            state = State(prev_team=team, wealth=W + float(moves[list(team)].sum()),
                          free_transfers=rules.free_per_round, chips_left=state.chips_left)
        else:
            team, boost, chip = best.teams[0], best.boosts[0], best.chips[0]
            b3 = best.boosts3[:1] if best.boosts3 else None
            if ap_left and chip is None:
                gains = [(FP.autopilot_gain(best.teams[j], pts[j]), -j) for j in range(Hh)
                         if best.chips[j] is None and j != k_lim]
                if max(gains)[1] == 0:               # this week is the window's best Autopilot week
                    chip, ap_left = "autopilot", False
                    boost = max((e for e in team if FP.IS_DRIVER[e]), key=lambda e: P[t][e])
            r, per, pen, trs, ok = evaluate_plan([team], [boost], [None if chip == "autopilot" else chip],
                                                 P[t][None, :], C[t][None, :], FP.IS_DRIVER, rules, state,
                                                 plan_boosts3=b3)
            assert ok, f"gameday {t + 1}: infeasible move"
            label = FP.ENTITIES[boost] if chip != "3x_booster" else f"{FP.ENTITIES[b3[0]]} 3x + {FP.ENTITIES[boost]} 2x"
            row = {"chip": chip or "", "transfers": trs[0], "penalty": pen[0], "team": team, "boost": label}
            used = 0 if (state.prev_team is None or chip == "wildcard") else min(trs[0], state.free_transfers)
            state = State(prev_team=team, wealth=W + float(moves[list(team)].sum()),
                          free_transfers=min(rules.max_bank, state.free_transfers - used + rules.free_per_round),
                          chips_left=tuple(c for c in state.chips_left if c != chip))
        total += r
        tm = row.pop("team")
        log.append({"gameday": t + 1, **row, "points": r, "cumulative": total,
                    "drivers": ", ".join(FP.ENTITIES[e] for e in sorted(tm) if FP.IS_DRIVER[e]),
                    "constructors": ", ".join(FP.ENTITIES[e] for e in sorted(tm) if not FP.IS_DRIVER[e])})
    return total, pd.DataFrame(log)


def load_forecasts(path, model=None):
    """{gameday: (rounds, planning-blend forecasts H x E)} from a season_replay_forecasts pickle, and the model's name."""
    res = pd.read_pickle(path)
    names = list(res[0][2])
    models = [n for n in names if n not in ("sim", "sim+ewma0.4")]
    model = model or models[-1]
    return {g: (list(rounds), 0.5 * out[model] + 0.5 * out["sim+ewma0.4"]) for g, rounds, out in res}, model


def _run(args):
    name, fc, rounds, P, C, kw = args
    total, log = play(fc, rounds, P, C, **kw)
    return name, total, log


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--top", type=float, help="the top player's season total, for comparison (e.g. 4994)")
    ap.add_argument("--forecasts", help="season_replay_forecasts_*.pkl (default: the newest in results/cache)")
    ap.add_argument("--model", help="forecaster in the pickle to blend (default: its model)")
    ap.add_argument("--workers", type=int, default=min(len(POLICIES), max(1, (os.cpu_count() or 2) // 2)))
    a = ap.parse_args()

    path = a.forecasts or (sorted(glob.glob(os.path.join("results", "cache", "season_replay_forecasts_*.pkl"))) or [None])[-1]
    if not path or not os.path.exists(path):
        raise SystemExit("no cached walk-forward forecasts: pass --forecasts PATH, or make them with "
                         "`python experiments/s26_season_replay.py results/s26_final_spec.json`")
    fcs, model = load_forecasts(path, a.model)
    gds, names, P, C = load_official()
    G = max(t for t in range(len(gds) + 1) if all(g in fcs for g in gds[:t]))
    if G < len(gds):
        print(f"note: the forecasts stop at gameday {gds[G - 1]}; scoring gamedays {gds[0]}-{gds[G - 1]} "
              f"(re-run experiments/s26_season_replay.py to add the newer ones)")
    P, C, gds, names = P[:G], C[:G], gds[:G], names[:G]
    fc = [fcs[g][1] for g in gds]
    rounds = [fcs[g][0] for g in gds]
    print(f"Walk-forward 2026, gamedays {gds[0]}-{gds[-1]} ({names[0]} .. {names[-1]}); forecasts {path} "
          f"(planning blend on {model})\n")

    jobs = [(n, fc, rounds, P, C, kw) for n, kw in POLICIES.items()]
    if a.workers == 1:
        out = list(map(_run, jobs))
    else:                                            # "spawn": forked workers can hang on HiGHS's thread locks
        with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("spawn")) as ex:
            out = list(ex.map(_run, jobs))

    opt = None
    if os.path.exists(OPTIMUM_CSV):
        o = pd.read_csv(OPTIMUM_CSV)
        if len(o) == G:
            opt = float(o["cumulative"].iloc[-1])
    rows = []
    for name, total, log in out:
        chips = ", ".join(f"{c} {gds[i]}" for i, c in enumerate(log["chip"]) if c)
        rows.append({"policy": name, "points": round(total), "chips (gameday)": chips,
                     **({"vs top player": round(total - a.top)} if a.top else {}),
                     **({"of the optimum": f"{total / opt:.0%}"} if opt else {})})
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 80)
    print(df.to_string(index=False))
    head = next(t for n, t, _ in out if n == HEADLINE)
    log = next(l for n, _, l in out if n == HEADLINE)
    print()
    if a.top:
        print(f"Top player: {a.top:.0f}.  The planner as deployed: {head:.0f} ({head - a.top:+.0f}, "
              f"{head / a.top:.0%} of the top score).")
    if opt:
        print(f"Best possible season with hindsight (season_optimum.py, Limitless included): {opt:.0f}"
              + (f"; the top player got {a.top / opt:.0%} of it, the planner {head / opt:.0%}." if a.top else "."))
    else:
        print("Run season_optimum.py for the ceiling (the best possible season, Limitless included).")
    print("The rows below the deployed planner are rules picked after seeing this season: treat their gains as "
          "optimistic.")
    log.insert(1, "race", names)
    print(f"\nThe planner as deployed, race by race:\n")
    print(log[["gameday", "race", "chip", "transfers", "penalty", "boost", "points", "cumulative"]]
          .round(1).to_string(index=False))
    os.makedirs("results", exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    log.to_csv(LOG_CSV, index=False)
    print(f"\nWrote {OUT_CSV} and {LOG_CSV}")


if __name__ == "__main__":
    main()
