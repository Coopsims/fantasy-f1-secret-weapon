#!/usr/bin/env python3
"""
Synthetic seasons with a *known* ground truth (f1_synthetic.py)
==============================================================
Used by the tests and by `f1_weekend_model.py demo` to check that the
ratings/calibration machinery recovers the truth and produces honest
uncertainty intervals - and to let the tool run end-to-end with no network.

The generating process is deliberately close to (but not identical to) the model:
car + driver pace, a shared team-mate shock, grid stickiness in the race,
reliability-driven DNFs and Poisson overtakes.
"""

import numpy as np
import pandas as pd

from f1_calendar import GRID_2026, HISTORICAL_CALENDARS, RACE_CALENDAR_2026, canon_location
from f1_weekend_data import Results, _finalise


def _price_prior():
    from f1_weekend_model import ModelParams, _prior_team_strength
    return _prior_team_strength(ModelParams())

SPRINT_LOCATIONS = {r["location"] for r in RACE_CALENDAR_2026 if r["sprint"]}


def synthetic_results(seed=0, years=(2024, 2025, 2026), rounds_2026=None,
                      c_true=1.8, rho_true=0.3, kappa_true=0.45, dnf_rate=0.07):
    rng = np.random.default_rng(seed)
    drivers = list(GRID_2026)
    teams = sorted(set(GRID_2026.values()))
    skill = dict(zip(drivers, rng.normal(0, 0.35, len(drivers))))
    team_strength = {}
    for y in years:
        s = rng.normal(0, 0.9, len(teams))
        if y == 2026:
            # new regulations: a reset, but the pre-season market view (prices) is
            # assumed to be partly right - the same assumption the model makes.
            prior = _price_prior()
            s = 0.6 * np.array([prior[t] for t in teams]) + rng.normal(0, 0.65, len(teams))
        elif y > min(years):                              # persistence inside a regulation set
            prev = np.array([team_strength[(t, y - 1)] for t in teams])
            s = 0.7 * prev + 0.3 * s
        for t, v in zip(teams, s):
            team_strength[(t, y)] = v
    team_dnf = {t: dnf_rate * rng.uniform(0.6, 1.5) for t in teams}

    rows, wk_rows = [], []
    truth = {"skill": skill, "team": team_strength, "c": c_true, "rho": rho_true}
    for y in years:
        cal = HISTORICAL_CALENDARS[y]
        n_rounds = len(cal) if not (y == 2026 and rounds_2026) else rounds_2026
        for rnd in range(1, n_rounds + 1):
            loc = canon_location(cal[rnd - 1])
            sprint = loc in SPRINT_LOCATIONS
            wk_rows.append({"year": y, "round": rnd, "order": rnd, "location": loc, "sprint": sprint})
            drift = {t: team_strength[(t, y)] + rng.normal(0, 0.05) * rnd ** 0.5 for t in teams}
            shock = {t: rng.normal() for t in teams}

            def perf(extra=None):
                out = {}
                for d in drivers:
                    t = GRID_2026[d]
                    base = drift[t] + skill[d] + (extra[d] if extra else 0.0)
                    out[d] = c_true * base + np.sqrt(rho_true) * shock[t] + np.sqrt(1 - rho_true) * rng.normal()
                return out

            def order(p):
                return {d: i + 1 for i, d in enumerate(sorted(p, key=lambda k: -p[k]))}

            def emit(code, finish, grid=None, dnf=(), fl=None, ot=None):
                for d in drivers:
                    rows.append({"year": y, "round": rnd, "location": loc, "session": code,
                                 "driver": d, "team": GRID_2026[d],
                                 "grid": grid[d] if grid else np.nan, "finish": finish[d],
                                 "dnf": d in dnf, "fastest_lap": d == fl,
                                 "overtakes": (ot or {}).get(d, np.nan), "n_cars": len(drivers)})

            q = order(perf())
            emit("Q", q)
            sessions = [("R", q, 1.0, 1.0)]
            if sprint:
                sq = order(perf())
                emit("SQ", sq)
                sessions.insert(0, ("S", sq, 1.15, 0.35))
            for code, grid, stick, dnf_mult in sessions:
                gz = {d: _z(grid[d], len(drivers)) for d in drivers}
                dnf = {d for d in drivers if rng.random() < team_dnf[GRID_2026[d]] * dnf_mult}
                p = perf({d: kappa_true * stick * gz[d] / c_true * c_true for d in drivers})
                # classified order; DNFs at the back
                fin = sorted([d for d in drivers if d not in dnf], key=lambda k: -p[k]) + \
                      sorted(dnf, key=lambda k: rng.random())
                finish = {d: i + 1 for i, d in enumerate(fin)}
                fl_pool = [d for d in fin if d not in dnf][:10]
                fl = fl_pool[rng.integers(len(fl_pool))] if code == "R" or True else None
                ot = {d: int(rng.poisson(1.5 + 0.8 * max(grid[d] - finish[d], 0))) for d in drivers}
                emit(code, finish, grid=grid, dnf=dnf, fl=fl, ot=ot)
    res = _finalise(rows, [], wk_rows)
    res.truth = truth
    return res


def _z(pos, n):
    from statistics import NormalDist
    return NormalDist().inv_cdf((n - pos + 0.5) / n)
