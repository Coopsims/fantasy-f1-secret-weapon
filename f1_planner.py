#!/usr/bin/env python3
"""
Season / next-round planner core (f1_planner.py)
================================================
1. Forecast every remaining 2026 race for all 22 drivers and 11 constructors with the selected model spec
   (tree model(s) on the zoo feature panel + the weekend simulator + recent official form, blended per the spec).
   Features for a future race are "as of now" (form, Elo, team form) plus that circuit's own track features; practice
   features are empty until the weekend's practice has run (re-run after FP for the next race).
2. Prices: the game's current prices (fantasy feed), projected forward with the game's price rule (f1_prices).
3. Solve exactly (f1_strategy MILP) for teams, boosts, transfers and chips over the remaining races from any state.

Used by plan_season.py (whole remaining season) and plan_next_race.py (next round).
"""
import json
import os
from dataclasses import replace

import numpy as np
import pandas as pd

from f1_calendar import GRID_2026, RACE_CALENDAR_2026, canon_driver, canon_team
from f1_strategy import Rules, State, evaluate_plan, solve_strategy
from f1_weekend_data import Results

DRIVERS = list(GRID_2026)
TEAMS = sorted(set(GRID_2026.values()))
ENTITIES = DRIVERS + TEAMS
IS_DRIVER = np.array([True] * len(DRIVERS) + [False] * len(TEAMS))
SPEC_FILE = os.path.join("results", "s26_final_spec.json")


def canon_entity(name):
    t = canon_team(name)
    return t if t in TEAMS else canon_driver(name)


def upcoming_rounds(res, n=None):
    done = set(res.weekends[res.weekends["year"] == 2026]["round"])
    rr = [r["round"] for r in RACE_CALENDAR_2026 if not r.get("cancelled") and r["round"] not in done]
    return rr[:n] if n else rr


def _future_truth_rows(truth, t, rnd):
    race = next(r for r in RACE_CALENDAR_2026 if r["round"] == rnd)
    rows = pd.DataFrame({"driver": DRIVERS, "team": [GRID_2026[d] for d in DRIVERS]})
    for c in truth.columns:
        if c not in rows:
            rows[c] = np.nan
    rows["t"], rows["year"], rows["round"] = t, 2026, rnd
    rows["location"], rows["is_sprint"], rows["race_dnf"] = race["location"], bool(race["sprint"]), False
    return rows[truth.columns]


def future_panel(res, truth, rounds, full=None, pdata=None, tracks=None):
    """Feature rows for each future round as if it were the next weekend (t = last + 1)."""
    from experiments import zoo
    t_next = int(res.weekends["t"].max()) + 1
    out = []
    for rnd in rounds:
        race = next(r for r in RACE_CALENDAR_2026 if r["round"] == rnd)
        wk = pd.concat([res.weekends, pd.DataFrame([{"year": 2026, "round": rnd, "order": int((res.weekends.year == 2026).sum()) + 1,
                                                       "location": race["location"], "sprint": bool(race["sprint"]), "t": t_next}])],
                       ignore_index=True)
        res_p = Results(res.sessions, res.pits, wk)
        tr_p = pd.concat([truth, _future_truth_rows(truth, t_next, rnd)], ignore_index=True)
        p = zoo.build_panel(res_p, tr_p, full=full, pdata=pdata, tracks=tracks)
        out.append(p[p["t"] == t_next].assign(round=rnd))
    return pd.concat(out, ignore_index=True)


def fit_tree(spec, panel_hist, t_now, official=None):
    """Fit one spec's tree model on all completed weekends (weights as in experiments/s26.predict_spec)."""
    from experiments import zoo
    from experiments.s26 import _fit
    feats = zoo.FEATURES[spec["fset"]] + ["regime"]
    tr = panel_hist[panel_hist["d_last1"].notna()]
    y = tr["total"].values.copy()
    if spec.get("use_official") and "official" in tr:
        y = np.where(tr["official"].isna(), y, tr["official"])
    age = t_now - tr["t"].values
    w = 0.5 ** (age / spec["half_life"]) if spec["half_life"] < 999 else np.ones(len(tr))
    w = w * np.where(tr["regime"].values > 0, spec["w26"], 1.0)
    m = zoo.make_model(spec["family"], spec["hp"], seed=spec.get("seed", 0))
    _fit(m, tr[feats], y, w)
    return m, feats


def sim_forecasts(res, rounds, n=6000):
    """Simulator expected points (incl. Driver of the Day for drivers) for each future round, in ENTITIES order."""
    from f1_weekend_model import WeekendForecaster
    f = WeekendForecaster(res, year=2026, verbose=False)
    out = {}
    for r in rounds:
        fc = f.forecast(r, n=n, seed=100 + r)
        d = dict(zip(fc.drivers, fc.total_points(include_dotd=True).mean(0)))
        c = dict(zip(fc.teams, fc.constructor_total().mean(0)))
        out[r] = np.array([d.get(e, 0.0) for e in DRIVERS] + [c.get(e, 0.0) for e in TEAMS])
    return out


def official_tables(path="fantasy_f1_official_points.csv"):
    op = pd.read_csv(path)
    return op


def ewma_official(op, alpha):
    """Recent form: EWMA of each asset's official points over the gamedays it took part in (latest included)."""
    w = op.pivot_table(index="gameday", columns="name", values="points").reindex(columns=ENTITIES).sort_index()
    return w.ewm(alpha=alpha, adjust=False, ignore_na=True).mean().iloc[-1]


def constructor_bonus(op, alpha=0.3):
    """Official constructor points minus its two drivers' official points (quali bonus, pit stops ...), EWMA."""
    d = op[op.type == "driver"].assign(team=lambda x: x["name"].map(GRID_2026)).groupby(["gameday", "team"])["points"].sum()
    c = op[op.type == "constructor"].set_index(["gameday", "name"])["points"]
    extra = (c - d.reindex(c.index).values).rename("extra").reset_index().rename(columns={"name": "team"})
    out = {}
    for team, g in extra.sort_values("gameday").groupby("team"):
        out[team] = float(g["extra"].ewm(alpha=alpha, adjust=False).mean().iloc[-1])
    return out


def load_spec(path=SPEC_FILE):
    with open(path) as f:
        return json.load(f)


def forecast_remaining(res, truth, specs, rounds, panel_hist, full=None, pdata=None, tracks=None, sim=None,
                       op=None, c_sim=0.5, verbose=True, bonus_alpha=0.3, planning_blend=None):
    """
    Expected points (H x E) for the given future rounds under the blended spec(s):
      drivers      = mean over specs of [(1 - b - c) * tree + b * simulator + c * EWMA(official)]  (b from the spec's schedule)
      constructors = c_sim * simulator + (1 - c_sim) * (sum of its drivers' forecasts + recent official bonus)
    """
    op = official_tables() if op is None else op
    sim = sim or sim_forecasts(res, rounds)
    fut = future_panel(res, truth, rounds, full=full, pdata=pdata, tracks=tracks)
    fut["regime"] = 1.0
    t_now = int(res.weekends["t"].max()) + 1
    n26 = int((res.weekends["year"] == 2026).sum())
    drv = np.zeros((len(rounds), len(DRIVERS)))
    fallback = fut[fut["round"] == rounds[0]].set_index("driver").reindex(DRIVERS)["d_ewm"]   # reconstructed-points form
    for spec in specs:
        m, feats = fit_tree(spec, panel_hist, t_now)
        if len(op):
            ew = ewma_official(op, spec["ewma_alpha"]).reindex(ENTITIES)
        else:
            ew = pd.Series(np.nan, index=ENTITIES)
        ew[DRIVERS] = ew[DRIVERS].fillna(fallback)          # as in experiments/s26.prepare
        if "b_early" in spec:
            b = spec["b_late"] + (spec["b_early"] - spec["b_late"]) * np.exp(-n26 / spec["kappa"])
        else:
            b = spec.get("b_sim", 0.0)
        c = spec["c_ewma"]
        for k, r in enumerate(rounds):
            rows = fut[fut["round"] == r].set_index("driver").reindex(DRIVERS)
            tree = m.predict(rows[feats]).astype(float)
            s = sim[r][:len(DRIVERS)]
            e = np.nan_to_num(ew[DRIVERS].values, nan=0.0)
            drv[k] += ((1 - b - c) * tree + b * s + c * e) / len(specs)
    bonus = constructor_bonus(op, bonus_alpha) if len(op) else {}
    out = np.zeros((len(rounds), len(ENTITIES)))
    out[:, :len(DRIVERS)] = drv
    for k, r in enumerate(rounds):
        for j, team in enumerate(TEAMS):
            derived = sum(drv[k][i] for i, d in enumerate(DRIVERS) if GRID_2026[d] == team) + bonus.get(team, 0.0)
            out[k, len(DRIVERS) + j] = c_sim * sim[r][len(DRIVERS) + j] + (1 - c_sim) * derived
    if planning_blend:
        # Shrink for DECISIONS: the optimiser chases the biggest predicted edges, so noisy differences get over-acted
        # on; mixing in the simulator and recent official form (all 33 assets) played the 2026 season best (4151 vs
        # 3953 for the model alone; in-sample, one season) at a small cost in RMSE (+0.16, CI incl. 0).
        pb = planning_blend
        ew = ewma_official(op, pb.get("ewma_alpha", 0.2)).reindex(ENTITIES) if len(op) else pd.Series(np.nan, index=ENTITIES)
        for k, r in enumerate(rounds):
            e = ew.values.astype(float)
            e = np.where(np.isnan(e), sim[r], e)
            out[k] = pb["w_model"] * out[k] + pb["w_sim"] * sim[r] + pb["w_ewma"] * e
    return out


def current_prices(path="fantasy_f1_prices_official.csv", overrides=None):
    pr = pd.read_csv(path)
    cur = pr[pr["gameday"] == pr["gameday"].max()].set_index("name")["price"]
    price = np.array([float(cur[e]) for e in ENTITIES])
    for name, v in (overrides or {}).items():
        price[ENTITIES.index(canon_entity(name))] = float(v)
    return price


def project_prices(price_now, pts_fc, op, prices_path="fantasy_f1_prices_official.csv"):
    """Price path over the horizon with the game's rule, starting from the last two raced weekends' real PPM."""
    from f1_prices import PriceRule, step_prices
    rule = PriceRule()
    pr = pd.read_csv(prices_path)
    g_last = int(op["gameday"].max())
    past = []
    for g in (g_last - 1, g_last):
        p = pr[pr.gameday == g].set_index("name")["price"].reindex(ENTITIES).values
        pts = op[op.gameday == g].set_index("name")["points"].reindex(ENTITIES).fillna(0).values
        past.append(pts / np.maximum(p, 0.1))
    P = [np.asarray(price_now, float)]
    ppm = list(past)
    for k in range(len(pts_fc)):
        ppm.append(pts_fc[k] / np.maximum(P[-1], 0.1))
        P.append(step_prices(P[-1], ppm[-rule.window:], rule))
    return np.array(P)


ERROR_SD = {"driver": 14.6, "constructor": 20.6}       # held-out 2026 RMSE vs official points (results/BEST_MODEL_2026.md)


def _sigma():
    return np.where(IS_DRIVER, ERROR_SD["driver"], ERROR_SD["constructor"])


def nn_gain(pts):
    """E[max(X, 0)] - E[X] for X ~ Normal(forecast, error sd): what No Negative is worth per held asset."""
    from scipy.stats import norm
    mu, sd = np.asarray(pts, float), _sigma()[None, :]
    z = mu / sd
    return sd * norm.pdf(z) - mu * (1 - norm.cdf(z))


def autopilot_gain(team, pts_row, n=20000, seed=0):
    """Autopilot doubles the team's top-scoring driver after the race: E[max X_i] - max E[X_i] over the team's drivers."""
    d = [e for e in team if IS_DRIVER[e]]
    rng = np.random.default_rng(seed)
    draws = np.asarray(pts_row)[d][None, :] + ERROR_SD["driver"] * rng.standard_normal((n, len(d)))
    return float(draws.max(1).mean() - np.max(np.asarray(pts_row)[d]))


def limitless_team(pts_row, locked_out=()):
    """Best unconstrained team for one race (no budget, no transfer cost): top 5 drivers + top 2 constructors, 2x boost."""
    v = np.array(pts_row, float)
    for n in locked_out:
        v[ENTITIES.index(canon_entity(n))] = -1e9
    d = np.argsort(-np.where(IS_DRIVER, v, -np.inf))[:5]
    c = np.argsort(-np.where(~IS_DRIVER, v, -np.inf))[:2]
    boost = d[0]
    return frozenset(d.tolist() + c.tolist()), int(boost), float(v[d].sum() + v[c].sum() + v[boost])


def plan(pts, price_path, team=None, bank=0.0, budget=100.0, free=2, chips=(), discount=1.0, rules=None,
         locked_in=(), locked_out=(), time_limit=180):
    """
    Exact plan over len(pts) races from the given state.  team=None -> fresh team with `budget`.
    'limitless' in chips: every possible Limitless week is tried (that week the best unconstrained team scores and the
    real team is parked unchanged, as the game reverts it), plus not using it; the best total wins.
    Returns (Plan, State); plan.chips[k] == 'limitless' marks the week, plan.teams[k] is then the Limitless team.
    """
    chips = tuple(c for c in chips if c in ("wildcard", "3x_booster", "no_negative", "limitless"))   # autopilot / final_fix: reported, not optimised
    if "limitless" in chips:
        rest = tuple(c for c in chips if c != "limitless")
        best = plan(pts, price_path, team, bank, budget, free, rest, discount, rules, locked_in, locked_out, time_limit)
        H = len(pts)
        for k in range(H):
            if k == 0 and not team:
                continue
            p_k = np.array(pts, float)
            lt, lb, lv = limitless_team(p_k[k], locked_out)
            p_k[k] = 0.0                                    # the parked real team scores nothing that week
            cand, st = _plan(p_k, price_path, team, bank, budget, free, rest, discount, rules, locked_in, locked_out,
                             time_limit, _frozen=(k,))
            if cand.total is None or not np.isfinite(cand.total):
                continue
            tot = cand.total + (discount ** k) * lv
            if tot > best[0].total + 1e-9:
                teams, boosts, chs, rp = list(cand.teams), list(cand.boosts), list(cand.chips), list(cand.round_points)
                teams[k], boosts[k], chs[k], rp[k] = lt, lb, "limitless", lv
                best = (replace(cand, teams=teams, boosts=boosts, chips=chs, round_points=rp, total=tot), st)
        return best
    return _plan(pts, price_path, team, bank, budget, free, chips, discount, rules, locked_in, locked_out, time_limit)


def _plan(pts, price_path, team=None, bank=0.0, budget=100.0, free=2, chips=(), discount=1.0, rules=None,
          locked_in=(), locked_out=(), time_limit=180, _frozen=()):
    rules = rules or Rules()
    H = len(pts)
    if team:
        idx = frozenset(ENTITIES.index(canon_entity(n)) for n in team)
        wealth = float(price_path[0][list(idx)].sum()) + bank
        state = State(prev_team=idx, wealth=wealth, free_transfers=free, chips_left=tuple(chips))
    else:
        state = State(prev_team=None, wealth=budget, free_transfers=free, chips_left=tuple(chips))
    pts = np.array(pts, float)
    big = 1e4
    for n in locked_out:                                   # never pick these
        pts[:, ENTITIES.index(canon_entity(n))] = -big
    for n in locked_in:                                    # always keep these
        pts[:, ENTITIES.index(canon_entity(n))] += big
    disc = discount ** np.arange(H)
    p = solve_strategy(pts, price_path[:H], IS_DRIVER, rules, state, disc, time_limit=time_limit, mip_rel_gap=0.0,
                       frozen=_frozen, nn_gain=nn_gain(pts))
    if locked_in or locked_out:                            # report true expected points, not the steering offsets
        true = np.array(pts, float)
        for n in locked_out:
            true[:, ENTITIES.index(canon_entity(n))] += big
        for n in locked_in:
            true[:, ENTITIES.index(canon_entity(n))] -= big
        tot, per, pen, trs, ok = evaluate_plan(p.teams, p.boosts, p.chips, true, price_path[:H], IS_DRIVER, rules, state)
        p = replace(p, round_points=list(per), total=float(tot))
    return p, state


# ──────────────────────────────────────────────────────────────
# MLflow model (registered by experiments/s26_register.py)
# ──────────────────────────────────────────────────────────────

try:
    import mlflow.pyfunc as _pyfunc

    class FantasyPointsModel(_pyfunc.PythonModel):
        """
        Driver fantasy-points forecaster for 2026 (the selected spec ensemble).
        Input: one row per driver-weekend with the zoo feature columns, 'regime', 'sim' (simulator expected points incl.
        Driver of the Day) and 'ew_off_<alpha>' (EWMA of official points) - as built by experiments/s26.prepare or
        f1_planner.future_panel.  Param 'n26' = number of 2026 weekends already raced (sets the simulator weight).
        Output: expected official fantasy points per row.  Constructors: sum of the two drivers + recent bonus (c_sim 0).
        """
        def __init__(self, models, feats, specs):
            self.models, self.feats, self.specs = models, feats, specs

        def predict(self, context, model_input, params=None):
            n26 = float((params or {}).get("n26", 17))
            out = np.zeros(len(model_input))
            for m, f, sp in zip(self.models, self.feats, self.specs):
                tree = m.predict(model_input[f]).astype(float)
                b = sp["b_late"] + (sp["b_early"] - sp["b_late"]) * np.exp(-n26 / sp["kappa"]) if "b_early" in sp else sp.get("b_sim", 0.0)
                sim = model_input["sim"].fillna(pd.Series(tree, index=model_input.index)).values if "sim" in model_input else tree
                ew = model_input.get(f"ew_off_{sp['ewma_alpha']}", pd.Series(tree, index=model_input.index)).fillna(0).values
                out += ((1 - b - sp["c_ewma"]) * tree + b * sim + sp["c_ewma"] * ew) / len(self.models)
            return out
except ImportError:            # mlflow not installed: the planners still work
    pass


def autopilot_advice(plan, pts):
    """Best week for Autopilot on the planned teams (weeks without another chip). Returns (week index, gain) or None."""
    best = None
    for k, t in enumerate(plan.teams):
        if plan.chips[k]:
            continue
        g = autopilot_gain(t, pts[k])
        if best is None or g > best[1]:
            best = (k, g)
    return best
