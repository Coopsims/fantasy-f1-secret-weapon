#!/usr/bin/env python3
"""
Fantasy F1 scoring rules (f1_rules.py)
======================================
Single source of truth for the scoring constants and for *vectorised*
scoring functions.  Everything here is pure numpy so it can be used both on
one real result and on a [n_sims, n_drivers] block of simulated results.

No heavy imports (no sklearn / xgboost / network) on purpose.
"""

import numpy as np

# ──────────────────────────────────────────────────────────────
# Fantasy F1 2026 Scoring Rules
# ──────────────────────────────────────────────────────────────

# Qualifying points (driver)
QUALI_PTS = {1: 10, 2: 9, 3: 8, 4: 7, 5: 6, 6: 5, 7: 4, 8: 3, 9: 2, 10: 1}

# Race finish points (driver)
RACE_PTS = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}

# Sprint finish points (driver)
SPRINT_PTS = {1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1}

# Penalties and bonuses (driver)
DNF_PENALTY_RACE = -20
DNF_PENALTY_SPRINT = -10
DSQ_PENALTY = -20
QUALI_NC_PENALTY = -5
OVERTAKE_BONUS = 1          # per position overtaken in race
POSITION_GAINED = 1         # per grid position gained (quali -> race)
POSITION_LOST = -1          # per grid position lost (quali -> race)
FASTEST_LAP_RACE = 10
FASTEST_LAP_SPRINT = 5
DRIVER_OF_THE_DAY = 10      # not in OpenF1 -> only ever *forecast*, never "actual"

# Constructor scoring
# Constructors score the combined total of both drivers' points
# (qualifying, race, sprint - but NOT Driver of the Day).
# Additional constructor-specific bonuses:
CONSTRUCTOR_QUALI_Q2_Q3 = {
    "neither_q2": -1,
    "one_q2": 1,
    "both_q2": 3,
    "one_q3": 5,
    "both_q3": 10,
}
CONSTRUCTOR_DSQ_PENALTY_QUALI = -5    # per disqualified driver in qualifying
CONSTRUCTOR_DSQ_PENALTY_SPRINT = -10  # per disqualified driver in sprint
CONSTRUCTOR_DSQ_PENALTY_RACE = -20    # per disqualified driver in race

# Qualifying knockout cut-offs for a 22-car grid:
# Q1 drops the slowest 6 (16 advance), Q2 drops 6 more (10 advance).
Q2_CUTOFF = 16
Q3_CUTOFF = 10

# Pit stop scoring (constructor, based on stationary stop time)
PIT_STOP_TIERS = [
    (0.0, 2.0, 20),    # under 2.0s
    (2.0, 2.2, 10),    # 2.00 - 2.19s
    (2.2, 2.5, 5),     # 2.20 - 2.49s
    (2.5, 3.0, 2),     # 2.50 - 2.99s
    (3.0, float('inf'), 0),  # over 3.0s
]
PIT_STOP_FASTEST_BONUS = 5     # fastest pit stop of the race
PIT_STOP_WORLD_RECORD_BONUS = 15  # new world record (current record 1.8s)
PIT_STOP_WORLD_RECORD_TIME = 1.8  # current world record in seconds


# ──────────────────────────────────────────────────────────────
# Vectorised helpers
# ──────────────────────────────────────────────────────────────

def _table(d, size=64):
    """Dict {position: pts} -> lookup array indexed by position (0 and >max -> 0)."""
    arr = np.zeros(size, dtype=float)
    for k, v in d.items():
        arr[k] = v
    return arr


_QUALI_T = _table(QUALI_PTS)
_RACE_T = _table(RACE_PTS)
_SPRINT_T = _table(SPRINT_PTS)


def _lookup(table, pos):
    pos = np.asarray(pos)
    idx = np.clip(pos.astype(int), 0, len(table) - 1)
    return table[idx]


def quali_points(pos):
    return _lookup(_QUALI_T, pos)


def race_finish_points(pos):
    return _lookup(_RACE_T, pos)


def sprint_finish_points(pos):
    return _lookup(_SPRINT_T, pos)


def position_change_points(grid, finish):
    """+1 per place gained, -1 per place lost (grid vs finish)."""
    gain = np.asarray(grid, dtype=float) - np.asarray(finish, dtype=float)
    return np.where(gain > 0, gain * POSITION_GAINED, (-gain) * POSITION_LOST)


def score_qualifying(pos, nc=None):
    """
    Driver qualifying points.  `nc` is an optional boolean mask (no time set /
    disqualified) that receives QUALI_NC_PENALTY instead of position points.
    """
    pts = quali_points(pos)
    if nc is not None:
        pts = np.where(nc, QUALI_NC_PENALTY, pts)
    return pts


def score_race_like(kind, grid, finish, dnf, overtakes, fastest_lap):
    """
    Score a race or sprint for drivers.  All inputs broadcast together.

    kind        : "race" or "sprint"
    grid        : starting position (1-based)
    finish      : finishing position (1-based; ignored where dnf)
    dnf         : boolean mask
    overtakes   : on-track passes
    fastest_lap : boolean mask

    Returns an ordered dict of component arrays.  Sum them for the event total.
    """
    if kind == "race":
        table, dnf_pen, fl_pts = _RACE_T, DNF_PENALTY_RACE, FASTEST_LAP_RACE
    elif kind == "sprint":
        table, dnf_pen, fl_pts = _SPRINT_T, DNF_PENALTY_SPRINT, FASTEST_LAP_SPRINT
    else:
        raise ValueError(kind)
    dnf = np.asarray(dnf, dtype=bool)
    finish_pts = np.where(dnf, 0.0, _lookup(table, finish))
    gain_pts = np.where(dnf, 0.0, position_change_points(grid, finish))
    return {
        "finish": finish_pts,
        "position_change": gain_pts,
        "overtakes": np.asarray(overtakes, dtype=float) * OVERTAKE_BONUS,
        "fastest_lap": np.where(np.asarray(fastest_lap, dtype=bool), float(fl_pts), 0.0),
        "dnf": np.where(dnf, float(dnf_pen), 0.0),
    }


def constructor_quali_bonus(pos_a, pos_b, nc_a=None, nc_b=None):
    """
    Team qualifying-progression bonus from both drivers' qualifying positions.
    NC drivers are treated as having missed Q2 (position = infinity).
    """
    pa = np.asarray(pos_a, dtype=float)
    pb = np.asarray(pos_b, dtype=float)
    if nc_a is not None:
        pa = np.where(nc_a, np.inf, pa)
    if nc_b is not None:
        pb = np.where(nc_b, np.inf, pb)
    q3 = (pa <= Q3_CUTOFF).astype(int) + (pb <= Q3_CUTOFF).astype(int)
    q2 = (pa <= Q2_CUTOFF).astype(int) + (pb <= Q2_CUTOFF).astype(int)
    B = CONSTRUCTOR_QUALI_Q2_Q3
    return np.where(
        q3 >= 2, B["both_q3"],
        np.where(q3 == 1, B["one_q3"],
                 np.where(q2 >= 2, B["both_q2"],
                          np.where(q2 == 1, B["one_q2"], B["neither_q2"]))),
    ).astype(float)


def pit_stop_points(times):
    """
    Points for an array of stationary stop times (NaN = no stop -> 0).
    Tiered by duration plus the world-record bonus.
    """
    t = np.asarray(times, dtype=float)
    pts = np.zeros_like(t)
    valid = ~np.isnan(t)
    for low, high, tier in PIT_STOP_TIERS:
        pts = np.where(valid & (t >= low) & (t < high), tier, pts)
    pts = np.where(valid & (t < PIT_STOP_WORLD_RECORD_TIME), pts + PIT_STOP_WORLD_RECORD_BONUS, pts)
    return pts


def event_total(components):
    """Sum a component dict into a single array."""
    out = 0.0
    for v in components.values():
        out = out + v
    return out
