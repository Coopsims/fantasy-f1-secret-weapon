#!/usr/bin/env python3
"""
F1 Fantasy weekend simulator (f1_weekend_model.py)
==================================================
Predicts every *scoring event* of a race weekend - Qualifying, Sprint
(+ sprint grid), Race - for every driver and constructor, as full probability
distributions rather than one number.

How it works
------------
1. RATINGS  Past sessions are turned into normal-score outcomes and fitted with a
   time-decayed ridge regression:   y = team-season effect + driver effect
   (+ grid effect for races).  Qualifying and race pace are rated separately;
   pre-2026 team effects are nuisance terms so only *driver* skill carries over
   the regulation reset, while 2026 teams start from a prior (default: the
   pre-season constructor prices) and are re-learned as real rounds arrive.
2. CALIBRATION  A walk-forward pass (fit on the past, predict the next weekend)
   calibrates how noisy a session is (scale `c`) and how correlated team-mates
   are (`rho`), so the simulated spread is honest, not a guess.
3. SIMULATION  Each weekend is simulated thousands of times:
   qualifying order -> sprint grid -> sprint -> race grid -> DNFs -> finish order
   -> overtakes -> fastest lap -> pit stops -> Driver of the Day.
   Every Monte-Carlo draw is scored with the exact fantasy rules (f1_rules.py),
   giving per-event, per-component points with means, spreads and quantiles.
4. CONDITIONING  Real results can be fed in as they happen (`known=`): after
   qualifying the race forecast is conditioned on the true grid, etc.

CLI
---
    python f1_weekend_model.py forecast --round 19
    python f1_weekend_model.py forecast --round 19 --quali "Norris,Piastri,..."
    python f1_weekend_model.py backtest --years 2025
    python f1_weekend_model.py tune --years 2025
    python f1_weekend_model.py demo            # runs end-to-end on synthetic data
"""

import argparse
import json
import math
from dataclasses import dataclass, field, replace
from statistics import NormalDist

import numpy as np
import pandas as pd

import f1_rules as R
from f1_calendar import (GRID_2026, get_race_info, track_profile, RACE_CALENDAR_2026,
                         DEFAULT_CONSTRUCTOR_PRICES)
from f1_weekend_data import Results, load_results, score_weekend_actual

_ND = NormalDist()
REGIME_START_YEARS = (2026,)       # years where the technical regulations reset

# Pre-season constructor prices: a market-implied prior for 2026 car strength.
PRIOR_CONSTRUCTOR_PRICES = DEFAULT_CONSTRUCTOR_PRICES

# Relative DNF propensity of each 2026 team before any data (new power units!).
RELIABILITY_PRIOR = {
    "Mercedes": 0.9, "Ferrari": 0.95, "McLaren": 0.9, "Red Bull Racing": 1.15,
    "Racing Bulls": 1.15, "Audi": 1.45, "Cadillac": 1.5, "Aston Martin": 1.3,
    "Williams": 1.0, "Alpine": 1.0, "Haas F1 Team": 1.05,
}

# Prior mean stationary stop time (s) per team before any pit data.
PIT_PRIOR = {
    "McLaren": 2.30, "Red Bull Racing": 2.35, "Mercedes": 2.45, "Ferrari": 2.50,
    "Williams": 2.55, "Racing Bulls": 2.60, "Aston Martin": 2.60, "Alpine": 2.65,
    "Haas F1 Team": 2.65, "Audi": 2.75, "Cadillac": 2.90,
}


# ──────────────────────────────────────────────────────────────
# Parameters
# ──────────────────────────────────────────────────────────────

@dataclass
class ModelParams:
    half_life: float = 10.0        # weekends until a session's weight halves
    driver_ridge: float = 4.0      # shrinkage of driver skill toward 0
    team_ridge_new: float = 2.0    # shrinkage of target-year team strength to its prior
    team_ridge_old: float = 0.3    # nuisance team-season effects
    team_link: float = 2.0         # persistence between consecutive same-regulation seasons
    link_rho: float = 0.8
    sprint_weight: float = 0.6     # weight of sprint sessions in rating fits
    prior_scale: float = 0.8       # price z-score -> rating units
    n_calib: int = 24              # weekends used for walk-forward calibration
    min_calib_history: int = 8     # weekends of history needed before calibrating
    default_c: float = 1.6         # fallbacks / shrinkage targets
    default_rho: float = 0.25
    weekend_shock: float = 0.4     # share of team shock common to all sessions of a weekend
    p_quali_nc: float = 0.012      # no-time / DSQ in qualifying
    sprint_dnf_factor: float = 0.35
    chaos_sigma: float = 0.35      # weekend-level multiplier on DNF probability
    fl_beta: float = 0.6           # fastest-lap preference for pace
    dotd: bool = True
    # -- opt-in extensions (defaults reproduce the original model exactly) --
    team_block: int = 0            # >0: team strength may drift every `team_block` weekends within a season
    block_link: float = 4.0        # random-walk stiffness between consecutive blocks (higher = smoother)
    p0_min: float = 0.03           # clip range of the base DNF rate
    p0_max: float = 0.15
    dnf_a_team: float = 20.0       # pseudo-starts shrinking team DNF rate to its prior
    dnf_a_driver: float = 5.0      # pseudo-DNFs shrinking driver multiplier to 1
    dnf_half_life_mult: float = 2.0


# ──────────────────────────────────────────────────────────────
# Small numerics
# ──────────────────────────────────────────────────────────────

def _erf(x):
    x = np.asarray(x, dtype=float)
    s, a = np.sign(x), np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * a)
    poly = ((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592
    return s * (1.0 - poly * t * np.exp(-a * a))


def _phi(x):
    return 0.5 * (1.0 + _erf(np.asarray(x) / math.sqrt(2.0)))


_Z_CACHE = {}


def _zscore(rank, n):
    """Normal score for finishing `rank` (1=best) out of n: higher is better."""
    key = (int(rank), int(n))
    v = _Z_CACHE.get(key)
    if v is None:
        v = _ND.inv_cdf((n - rank + 0.5) / n)
        _Z_CACHE[key] = v
    return v


def _rank_positions(perf):
    """perf (n, k) higher=better -> positions 1..k (n, k)."""
    order = np.argsort(-perf, axis=1)
    pos = np.empty_like(order)
    np.put_along_axis(pos, order, np.arange(1, perf.shape[1] + 1)[None, :].repeat(perf.shape[0], 0), axis=1)
    return pos


def _regime(year):
    return sum(1 for y in REGIME_START_YEARS if year >= y)


def _prior_team_strength(params):
    p = pd.Series(PRIOR_CONSTRUCTOR_PRICES, dtype=float)
    z = (p - p.mean()) / p.std()
    return (z * params.prior_scale).to_dict()


# ──────────────────────────────────────────────────────────────
# Rating fit (one per family: qualifying-like and race-like)
# ──────────────────────────────────────────────────────────────

def build_family_obs(sessions, family, sprint_weight):
    """
    Per-driver normal-score outcomes for a family.
      family "Q": Q + SQ, all cars.
      family "R": R + S, classified finishers only, with a grid-position covariate.
    """
    if sessions.empty:
        return pd.DataFrame(columns=["t", "year", "session", "driver", "team", "y", "g", "w0"])
    if family == "Q":
        s = sessions[sessions["session"].isin(["Q", "SQ"])].copy()
    else:
        s = sessions[sessions["session"].isin(["R", "S"]) & ~sessions["dnf"]].copy()
    if s.empty:
        return pd.DataFrame(columns=["t", "year", "session", "driver", "team", "y", "g", "w0"])
    s["rank"] = s.groupby(["t", "session"])["finish"].rank(method="first")
    s["m"] = s.groupby(["t", "session"])["finish"].transform("size")
    s["y"] = [_zscore(r, m) for r, m in zip(s["rank"], s["m"])]
    if family == "R":
        gv = s["grid"].where(s["grid"].notna(), np.nan)
        s["grank"] = gv.groupby([s["t"], s["session"]]).rank(method="first")
        s["g"] = [(_zscore(gr, m) if pd.notna(gr) else 0.0) for gr, m in zip(s["grank"], s["m"])]
    else:
        s["g"] = 0.0
    s["w0"] = np.where(s["session"].isin(["SQ", "S"]), sprint_weight, 1.0)
    return s[["t", "year", "session", "driver", "team", "y", "g", "w0"]].reset_index(drop=True)


@dataclass
class FamilyFit:
    drivers: list
    team_years: list
    beta: np.ndarray
    cov: np.ndarray
    kappa: float
    sigma2: float
    with_grid: bool
    target_year: int
    target_blk: int = 0

    def _index(self, driver, team):
        d = self.drivers.index(driver) if driver in self.drivers else None
        k = (team, self.target_year, self.target_blk)
        ty = self.team_years.index(k) if k in self.team_years else None
        return d, (len(self.drivers) + ty) if ty is not None else None

    def lp(self, field_):
        """
        Linear predictor (rating without the grid term) and its posterior variance
        for every (driver -> team) in `field_`, in field order.
        """
        mu, var = [], []
        nb = len(self.beta)
        for driver, team in field_.items():
            di, ti = self._index(driver, team)
            a = np.zeros(nb)
            if di is not None:
                a[di] = 1
            if ti is not None:
                a[ti] = 1
            mu.append(float(a @ self.beta))
            var.append(float(a @ self.cov @ a))
        return np.array(mu), np.array(var)


def fit_family(obs, now_t, target_year, field_, params, prior_team, with_grid):
    """Time-decayed ridge fit.  `field_` = {driver: team} for the target weekend."""
    if obs.empty:
        drivers = sorted(field_)
        team_years = sorted({(t, target_year, 0) for t in field_.values()})
        p = len(drivers) + len(team_years) + (1 if with_grid else 0)
        beta = np.zeros(p)
        for i, ty in enumerate(team_years):
            beta[len(drivers) + i] = prior_team.get(ty[0], 0.0) if _regime(target_year) >= 1 else 0.0
        lam = np.r_[np.full(len(drivers), params.driver_ridge),
                    np.full(len(team_years), params.team_ridge_new), [1e-6] * with_grid]
        return FamilyFit(drivers, team_years, beta, np.diag(1.0 / lam) * 0.6, 0.0, 0.6,
                         with_grid, target_year)

    w = obs["w0"].values * 0.5 ** ((now_t - obs["t"].values) / params.half_life)
    drivers = sorted(set(obs["driver"]) | set(field_))
    B = params.team_block
    ystart = obs.groupby("year")["t"].min().to_dict()
    ystart.setdefault(target_year, now_t)

    def blk_of(t_, y_):
        return int((t_ - ystart[y_]) // B) if B and B > 0 else 0

    obs_blk = [blk_of(t_, y_) for t_, y_ in zip(obs["t"].values, obs["year"].values)]
    target_blk = blk_of(now_t, target_year)
    ty_set = set(zip(obs["team"], obs["year"], obs_blk)) | {(t, target_year, target_blk) for t in field_.values()}
    team_years = sorted(ty_set, key=lambda k: (k[1], k[2], k[0]))
    d_idx = {d: i for i, d in enumerate(drivers)}
    ty_idx = {k: len(drivers) + i for i, k in enumerate(team_years)}
    nd, nty = len(drivers), len(team_years)
    p = nd + nty + (1 if with_grid else 0)

    n = len(obs)
    X = np.zeros((n, p))
    ri = np.arange(n)
    X[ri, [d_idx[d] for d in obs["driver"]]] = 1.0
    X[ri, [ty_idx[k] for k in zip(obs["team"], obs["year"], obs_blk)]] = 1.0
    if with_grid:
        X[:, -1] = obs["g"].values
    y = obs["y"].values
    Xw = X * w[:, None]
    A = Xw.T @ X
    b = Xw.T @ y

    lam = np.zeros(p)
    lam[:nd] = params.driver_ridge
    b0 = np.zeros(p)
    use_prior = _regime(target_year) >= 1
    first_new = {}
    for (team, yr, blk) in ty_idx:
        if yr == target_year:
            first_new[team] = min(first_new.get(team, blk), blk)
    for (team, yr, blk), j in ty_idx.items():
        if yr == target_year:
            if blk == first_new[team]:
                lam[j] = params.team_ridge_new
                b0[j] = prior_team.get(team, 0.0) if use_prior else 0.0
            else:
                lam[j] = 1e-3          # later blocks are tied to the previous one by the random-walk link
        else:
            lam[j] = params.team_ridge_old
    if with_grid:
        lam[-1] = 1e-6

    M = A + np.diag(lam)
    b = b + lam * b0
    # persistence between consecutive seasons inside one regulation set
    by_team = {}
    for (team, yr, blk) in ty_idx:
        by_team.setdefault(team, []).append((yr, blk))
    for team, keys in by_team.items():
        keys.sort()
        for (pyr, pblk), (yr, blk) in zip(keys[:-1], keys[1:]):
            j, prev = ty_idx[(team, yr, blk)], ty_idx[(team, pyr, pblk)]
            v = np.zeros(p)
            if yr == pyr:                                   # drift inside a season
                v[j], v[prev] = 1.0, -1.0
                M += params.block_link * np.outer(v, v)
            elif yr == pyr + 1 and _regime(yr) == _regime(pyr):   # persistence across seasons
                v[j], v[prev] = 1.0, -params.link_rho
                M += params.team_link * np.outer(v, v)

    Minv = np.linalg.inv(M)
    beta = Minv @ b
    resid = y - X @ beta
    sigma2 = float(max(0.3, (w * resid ** 2).sum() / max(w.sum(), 1e-9)))
    kappa = float(beta[-1]) if with_grid else 0.0
    return FamilyFit(drivers, team_years, beta, sigma2 * Minv, kappa, sigma2, with_grid, target_year, target_blk)


def _lp_for_rows(fit, rows, with_grid):
    """Out-of-sample linear predictor (incl. grid effect) for observation rows."""
    out = np.zeros(len(rows))
    for i, (driver, team, g) in enumerate(zip(rows["driver"], rows["team"], rows["g"])):
        di, ti = fit._index(driver, team)
        v = 0.0
        if di is not None:
            v += fit.beta[di]
        if ti is not None:
            v += fit.beta[ti]
        if with_grid:
            v += fit.kappa * g
        out[i] = v
    return out


@dataclass
class Calibration:
    c: float
    rho: float
    n_weekends: int


def calibrate_family(obs, weekends, params, prior_team, with_grid):
    """
    Walk-forward: for each of the most recent weekends, fit on the past only and
    see how well the predicted ratings order the cars.  Returns scale `c`
    (perf = c * rating + unit-variance noise) and team-mate correlation `rho`.
    """
    targets = [t for t in sorted(obs["t"].unique()) if (weekends["t"] < t).sum() >= params.min_calib_history]
    targets = targets[-params.n_calib:]
    deltas, outcomes, weights = [], [], []
    mates_a, mates_b, mate_w = [], [], []
    for t in targets:
        train = obs[obs["t"] < t]
        test = obs[obs["t"] == t]
        if train.empty or test.empty:
            continue
        year = int(test["year"].iloc[0])
        field_ = dict(zip(test["driver"], test["team"]))
        fit = fit_family(train, t, year, field_, params, prior_team, with_grid)
        lp = _lp_for_rows(fit, test, with_grid)
        test = test.assign(lp=lp, resid=test["y"].values - lp)
        for _, g in test.groupby("session"):
            n = len(g)
            if n < 4:
                continue
            ii, jj = np.triu_indices(n, 1)
            y = g["y"].values
            lpv = g["lp"].values
            tm = g["team"].values
            same = tm[ii] == tm[jj]
            keep = ~same
            d = (lpv[ii] - lpv[jj])[keep]
            o = (y[ii] > y[jj])[keep].astype(float)
            deltas.append(d)
            outcomes.append(o)
            weights.append(np.full(len(d), g["w0"].iloc[0]))
            res = g["resid"].values
            for a, b_ in zip(ii[same], jj[same]):
                mates_a.append(res[a])
                mates_b.append(res[b_])
                mate_w.append(g["w0"].iloc[0])
    if not deltas:
        return Calibration(params.default_c, params.default_rho, 0)

    d = np.concatenate(deltas)
    o = np.concatenate(outcomes)
    w = np.concatenate(weights)
    best_c, best_ll = params.default_c, -np.inf
    for c in np.linspace(0.3, 4.0, 75):
        p = np.clip(_phi(c * d / math.sqrt(2.0)), 1e-6, 1 - 1e-6)
        ll = float((w * (o * np.log(p) + (1 - o) * np.log(1 - p))).sum())
        if ll > best_ll:
            best_c, best_ll = float(c), ll

    ra, rb, rw = np.array(mates_a), np.array(mates_b), np.array(mate_w)
    if len(ra) >= 10:
        denom = (rw * (ra ** 2 + rb ** 2) / 2.0).sum()
        rho_hat = float((rw * ra * rb).sum() / denom) if denom > 0 else params.default_rho
    else:
        rho_hat = params.default_rho
    nw = len(targets)
    shrink = nw / (nw + 4.0)
    c = shrink * best_c + (1 - shrink) * params.default_c
    rho = shrink * rho_hat + (1 - shrink) * params.default_rho
    return Calibration(c, float(np.clip(rho, 0.0, 0.85)), nw)


# ──────────────────────────────────────────────────────────────
# DNF / overtakes / pit-stop sub-models
# ──────────────────────────────────────────────────────────────

class DnfModel:
    """Team reliability x driver incident multiplier, Bayesian-shrunk, regime-aware."""

    def __init__(self, sessions, now_t, target_year, params, a_team=None, a_driver=None):
        a_team = params.dnf_a_team if a_team is None else a_team
        a_driver = params.dnf_a_driver if a_driver is None else a_driver
        rr = sessions[sessions["session"] == "R"] if not sessions.empty else sessions
        self.p0 = 0.07
        self.team_p, self.driver_mult = {}, {}
        if rr is not None and len(rr) >= 40:
            w = 0.5 ** ((now_t - rr["t"].values) / (params.half_life * params.dnf_half_life_mult))
            w = w * np.where(rr["year"].map(_regime) == _regime(target_year), 1.0, 0.35)
            self.p0 = float(np.clip((w * rr["dnf"].values).sum() / w.sum(), params.p0_min, params.p0_max))
            rr = rr.assign(w=w)
            for team, g in rr.groupby("team"):
                prior = self.p0 * RELIABILITY_PRIOR.get(team, 1.1)
                self.team_p[team] = (float((g["w"] * g["dnf"]).sum()) + a_team * prior) / (g["w"].sum() + a_team)
            # driver multiplier = observed / expected-from-team
            exp_p = rr["team"].map(self.team_p).fillna(self.p0)
            for drv, g in rr.groupby("driver"):
                obs_k = float((g["w"] * g["dnf"]).sum())
                exp_k = float((g["w"] * exp_p.loc[g.index]).sum())
                # a_driver is in units of *expected DNFs* (~70 starts): drivers differ
                # in crash-proneness far less than cars differ in reliability.
                self.driver_mult[drv] = float(np.clip((obs_k + a_driver) / (exp_k + a_driver), 0.7, 1.5))

    def prob(self, driver, team):
        base = self.team_p.get(team, self.p0 * RELIABILITY_PRIOR.get(team, 1.1))
        return float(np.clip(base * self.driver_mult.get(driver, 1.0), 0.01, 0.4))


def fit_overtake_params(sessions):
    """a0, a1 in  E[overtakes] = a0 + a1 * places gained.  Falls back to real-world priors."""
    a0, a1 = 1.5, 0.8
    if sessions.empty:
        return a0, a1
    r = sessions[(sessions["session"] == "R") & ~sessions["dnf"]].dropna(subset=["grid", "overtakes"])
    if len(r) >= 100 and r["overtakes"].mean() > 0.5:
        gain = np.maximum(r["grid"].values - r["finish"].values, 0)
        X = np.c_[np.ones(len(r)), gain]
        coef, *_ = np.linalg.lstsq(X, r["overtakes"].values, rcond=None)
        a0, a1 = float(np.clip(coef[0], 0.3, 8.0)), float(np.clip(coef[1], 0.2, 1.6))
    return a0, a1


class PitModel:
    """Per-team stationary stop time (log-normal, asymmetric tails)."""

    SIGMA_LO, SIGMA_HI = 0.06, 0.10
    P_SLOW = 0.05

    def __init__(self, pits, now_t, params, a=10.0):
        self.mu = {t: math.log(s) for t, s in PIT_PRIOR.items()}
        if pits is not None and not pits.empty:
            p = pits[pits["stop_s"] < 5.0]
            w = 0.5 ** ((now_t - p["t"].values) / (params.half_life * 2.0))
            p = p.assign(w=w, lg=np.log(p["stop_s"].values))
            for team, g in p.groupby("team"):
                prior = self.mu.get(team, math.log(2.7))
                self.mu[team] = (float((g["w"] * g["lg"]).sum()) + a * prior) / (g["w"].sum() + a)

    def mu_for(self, team):
        return self.mu.get(team, math.log(2.7))


# ──────────────────────────────────────────────────────────────
# Fitted model bundle for a target weekend
# ──────────────────────────────────────────────────────────────

@dataclass
class FittedModel:
    field: dict
    year: int
    now_t: float
    q_fit: FamilyFit
    r_fit: FamilyFit
    q_cal: Calibration
    r_cal: Calibration
    dnf: DnfModel
    pit: PitModel
    ot: tuple
    params: ModelParams

    def summary(self):
        lq, _ = self.q_fit.lp(self.field)
        lr, _ = self.r_fit.lp(self.field)
        df = pd.DataFrame({"driver": list(self.field), "team": list(self.field.values()),
                           "quali_rating": lq, "race_rating": lr,
                           "dnf_prob": [self.dnf.prob(d, t) for d, t in self.field.items()]})
        return df.sort_values("race_rating", ascending=False).reset_index(drop=True)


def fit_model(res, field_, year, now_t, params=None, calib=None):
    """Fit all sub-models on `res` (which must contain only past weekends)."""
    params = params or ModelParams()
    prior_team = _prior_team_strength(params)
    sessions = res.sessions if not res.empty else pd.DataFrame()
    q_obs = build_family_obs(sessions, "Q", params.sprint_weight)
    r_obs = build_family_obs(sessions, "R", params.sprint_weight)
    q_fit = fit_family(q_obs, now_t, year, field_, params, prior_team, False)
    r_fit = fit_family(r_obs, now_t, year, field_, params, prior_team, True)
    if calib is None:
        wk = res.weekends if not res.empty else pd.DataFrame({"t": []})
        calib = (calibrate_family(q_obs, wk, params, prior_team, False) if len(q_obs) else
                 Calibration(params.default_c, params.default_rho, 0),
                 calibrate_family(r_obs, wk, params, prior_team, True) if len(r_obs) else
                 Calibration(params.default_c, params.default_rho, 0))
    return FittedModel(
        field=field_, year=year, now_t=now_t, q_fit=q_fit, r_fit=r_fit,
        q_cal=calib[0], r_cal=calib[1],
        dnf=DnfModel(sessions, now_t, year, params),
        pit=PitModel(res.pits if not res.empty else None, now_t, params),
        ot=fit_overtake_params(sessions), params=params,
    )


# ──────────────────────────────────────────────────────────────
# Forecast container
# ──────────────────────────────────────────────────────────────

EVENTS = ["quali", "sprint", "race"]


@dataclass
class WeekendForecast:
    drivers: list
    teams: list                 # constructor names
    team_of: list               # constructor index per driver
    race: dict                  # calendar info
    n: int
    sprint: bool
    pos: dict                   # name -> (n, nD) int positions (quali, sq, sprint, race)
    dnf: dict                   # "race"/"sprint" -> (n, nD) bool
    fl: dict                    # "race"/"sprint" -> (n, nD) bool
    comp: dict                  # (event, component) -> (n, nD) driver points
    dotd: np.ndarray            # (n, nD) driver-of-the-day points
    ccomp: dict                 # (event, component) -> (n, nT) constructor points
    meta: dict = field(default_factory=dict)

    # -- driver arrays ------------------------------------------------
    def event_points(self, event, include_dotd=True):
        parts = [v for (e, _), v in self.comp.items() if e == event]
        tot = np.sum(parts, axis=0) if parts else np.zeros((self.n, len(self.drivers)))
        if event == "race" and include_dotd:
            tot = tot + self.dotd
        return tot

    def total_points(self, include_dotd=True):
        return sum(self.event_points(e, include_dotd) for e in EVENTS)

    def constructor_event_points(self, event):
        parts = [v for (e, _), v in self.ccomp.items() if e == event]
        return np.sum(parts, axis=0) if parts else np.zeros((self.n, len(self.teams)))

    def constructor_total(self):
        return sum(self.constructor_event_points(e) for e in EVENTS)

    # -- convenience --------------------------------------------------
    def driver_scores(self, include_dotd=True):
        return dict(zip(self.drivers, np.round(self.total_points(include_dotd).mean(0), 1)))

    def constructor_scores(self):
        return dict(zip(self.teams, np.round(self.constructor_total().mean(0), 1)))

    def driver_std(self, include_dotd=True):
        return dict(zip(self.drivers, np.round(self.total_points(include_dotd).std(0), 1)))

    def driver_table(self):
        tot = self.total_points()
        rows = []
        for i, d in enumerate(self.drivers):
            q, r = self.pos["quali"][:, i], self.pos["race"][:, i]
            row = {
                "driver": d, "team": self.teams[self.team_of[i]],
                "total": tot[:, i].mean(), "sd": tot[:, i].std(),
                "p10": np.percentile(tot[:, i], 10), "p50": np.percentile(tot[:, i], 50),
                "p90": np.percentile(tot[:, i], 90),
                "quali": self.event_points("quali")[:, i].mean(),
                "sprint": self.event_points("sprint")[:, i].mean() if self.sprint else 0.0,
                "race": self.event_points("race")[:, i].mean(),
                "exp_quali_pos": q.mean(), "P_pole": (q == 1).mean(),
                "P_top10_q": (q <= 10).mean(),
                "exp_race_pos": float(r[~self.dnf["race"][:, i]].mean())
                if (~self.dnf["race"][:, i]).any() else np.nan,
                "P_win": ((r == 1) & ~self.dnf["race"][:, i]).mean(),
                "P_podium": ((r <= 3) & ~self.dnf["race"][:, i]).mean(),
                "P_points": ((r <= 10) & ~self.dnf["race"][:, i]).mean(),
                "P_dnf": self.dnf["race"][:, i].mean(),
                "P_fastest_lap": self.fl["race"][:, i].mean(),
            }
            rows.append(row)
        return pd.DataFrame(rows).sort_values("total", ascending=False).reset_index(drop=True)

    def constructor_table(self):
        tot = self.constructor_total()
        rows = []
        for j, t in enumerate(self.teams):
            row = {"team": t, "total": tot[:, j].mean(), "sd": tot[:, j].std(),
                   "p10": np.percentile(tot[:, j], 10), "p50": np.percentile(tot[:, j], 50),
                   "p90": np.percentile(tot[:, j], 90)}
            for e in EVENTS:
                row[e] = self.constructor_event_points(e)[:, j].mean()
            members = [i for i, ti in enumerate(self.team_of) if ti == j]
            if members:
                q = self.pos["quali"][:, members]
                row["P_both_q3"] = ((q <= R.Q3_CUTOFF).sum(1) == len(members)).mean()
                row["pit_pts"] = self.ccomp[("race", "pit")][:, j].mean()
            rows.append(row)
        return pd.DataFrame(rows).sort_values("total", ascending=False).reset_index(drop=True)

    def breakdown(self, name):
        """Mean points by event and component for a driver or constructor."""
        out = {}
        if name in self.drivers:
            i = self.drivers.index(name)
            for (e, c), v in self.comp.items():
                out.setdefault(e, {})[c] = float(v[:, i].mean())
            out.setdefault("race", {})["dotd"] = float(self.dotd[:, i].mean())
        elif name in self.teams:
            j = self.teams.index(name)
            for (e, c), v in self.ccomp.items():
                out.setdefault(e, {})[c] = float(v[:, j].mean())
        else:
            raise KeyError(name)
        return {e: {c: round(x, 2) for c, x in comps.items() if abs(x) > 1e-9 or c in ("finish",)}
                for e, comps in out.items() if e != "sprint" or self.sprint}

    # -- team-level risk ----------------------------------------------
    def team_samples(self, drivers, constructors, boost=None, boost_mult=2.0, no_negative=False):
        """Distribution of a fantasy team's weekend score (one value per simulation)."""
        pts = self.total_points()
        di = [self.drivers.index(d) for d in drivers]
        dp = pts[:, di].copy()
        if no_negative:
            dp = np.maximum(dp, 0.0)
        if boost in drivers:
            dp[:, drivers.index(boost)] *= boost_mult
        ci = [self.teams.index(c) for c in constructors]
        return dp.sum(1) + self.constructor_total()[:, ci].sum(1)

    def chip_analysis(self, drivers, constructors, boost=None):
        """Expected value of the boost choices and of the No-Negative chip for a team."""
        pts = self.total_points()
        out = {"boost": {}, "no_negative_gain": None}
        for d in drivers:
            i = self.drivers.index(d)
            x = pts[:, i]
            out["boost"][d] = {"x2_extra": float(x.mean()), "x3_extra": float(2 * x.mean()),
                               "sd": float(x.std()), "p_negative": float((x < 0).mean())}
        base = self.team_samples(drivers, constructors, boost)
        nn = self.team_samples(drivers, constructors, boost, no_negative=True)
        out["no_negative_gain"] = float((nn - base).mean())
        out["team_mean"] = float(base.mean())
        out["team_sd"] = float(base.std())
        out["team_p10_p90"] = (float(np.percentile(base, 10)), float(np.percentile(base, 90)))
        return out

    def to_dict(self):
        return {
            "race": self.race, "sprint": self.sprint, "n_sims": self.n,
            "drivers": json.loads(self.driver_table().round(3).to_json(orient="records")),
            "constructors": json.loads(self.constructor_table().round(3).to_json(orient="records")),
            "meta": self.meta,
        }


# ──────────────────────────────────────────────────────────────
# The simulator
# ──────────────────────────────────────────────────────────────

def _z_table(n):
    """z[pos-1] for a field of n (position 1 -> highest)."""
    return np.array([_zscore(p, n) for p in range(1, n + 1)])


def simulate_weekend(model, race_info, n=10000, seed=None, known=None,
                     grid_penalties=None, sprint=None):
    """
    Monte-Carlo a full weekend with a fitted model.

    known : optional dict of real results to condition on, each an ordered list of
            driver names (best first):  quali, sprint_quali, sprint.
            Drivers missing from a list are placed behind it by rating.
    grid_penalties : {driver: places} applied to the race grid after qualifying.
    """
    p = model.params
    rng = np.random.default_rng(seed)
    known = known or {}
    drivers = list(model.field)
    nD = len(drivers)
    teams = sorted(set(model.field.values()))
    team_of = np.array([teams.index(model.field[d]) for d in drivers])
    nT = len(teams)
    is_sprint = race_info.get("sprint", False) if sprint is None else sprint
    prof = track_profile(race_info.get("location", ""))

    lq, vq = model.q_fit.lp(model.field)
    lr, vr = model.r_fit.lp(model.field)
    cQ, cR = model.q_cal.c, model.r_cal.c
    rhoQ, rhoR = model.q_cal.rho, model.r_cal.rho
    kappa = float(np.clip(model.r_fit.kappa, 0.0, 1.5))
    zt = _z_table(nD)
    ω = p.weekend_shock

    # rating uncertainty beyond what calibration already averages in
    extra_q = np.sqrt(np.maximum(0.0, vq - np.median(vq)))
    extra_r = np.sqrt(np.maximum(0.0, vr - np.median(vr)))
    u_q = rng.standard_normal((n, nD)) * extra_q * cQ
    u_r = rng.standard_normal((n, nD)) * extra_r * cR

    W = rng.standard_normal((n, nT))     # weekend-level team shock shared across sessions

    def session_perf(base, c, rho, u, noise_mult=1.0):
        eta = np.sqrt(ω) * W + np.sqrt(1 - ω) * rng.standard_normal((n, nT))
        eps = rng.standard_normal((n, nD))
        return c * base[None, :] + u + noise_mult * (np.sqrt(rho) * eta[:, team_of] + np.sqrt(1 - rho) * eps)

    def known_positions(order):
        """List of names -> (n, nD) position array; unlisted drivers follow by rating."""
        names = [d for d in order if d in drivers]
        rest = sorted([d for d in drivers if d not in names], key=lambda d: -lr[drivers.index(d)])
        full = names + rest
        pos = np.empty(nD, dtype=int)
        for rank, d in enumerate(full, 1):
            pos[drivers.index(d)] = rank
        return np.broadcast_to(pos, (n, nD)).copy()

    def quali_session(key, known_key):
        if known_key in known:
            return known_positions(known[known_key]), np.zeros((n, nD), bool)
        perf = session_perf(lq, cQ, rhoQ, u_q)
        nc = rng.random((n, nD)) < p.p_quali_nc
        perf = np.where(nc, -1e9 + rng.random((n, nD)), perf)
        return _rank_positions(perf), nc

    q_pos, q_nc = quali_session("Q", "quali")
    if is_sprint:
        sq_pos, _ = quali_session("SQ", "sprint_quali")
    else:
        sq_pos = None

    def race_like(kind, grid, dnf_scale, stick_mult, ot_scale):
        """Simulate a race/sprint from a grid; returns pos, dnf, fl, overtakes, perf."""
        gz = zt[np.clip(grid, 1, nD) - 1]
        chaos_w = np.exp(p.chaos_sigma * rng.standard_normal((n, 1)) - p.chaos_sigma ** 2 / 2)
        base_p = np.array([model.dnf.prob(d, model.field[d]) for d in drivers])
        pd_ = np.clip(base_p[None, :] * prof["chaos"] * dnf_scale * chaos_w, 0.0, 0.6)
        dnf = rng.random((n, nD)) < pd_
        noise_mult = 1.0 + 0.3 * (prof["chaos"] - 1.0)
        base = lr[None, :] + kappa * prof["stickiness"] * stick_mult * gz
        eta = np.sqrt(ω) * W + np.sqrt(1 - ω) * rng.standard_normal((n, nT))
        eps = rng.standard_normal((n, nD))
        perf = cR * base + u_r + noise_mult * (np.sqrt(rhoR) * eta[:, team_of] + np.sqrt(1 - rhoR) * eps)
        eff = np.where(dnf, -1e9 + rng.random((n, nD)), perf)
        pos = _rank_positions(eff)
        gain = grid - pos
        a0, a1 = model.ot
        lam = prof["overtaking"] * ot_scale * (a0 + a1 * np.maximum(gain, 0))
        lam = np.where(dnf, 0.3 * a0 * prof["overtaking"] * ot_scale, lam)
        ot = rng.poisson(np.maximum(lam, 0))
        return pos, dnf, perf, ot

    def fastest_lap(pos, dnf, perf, beta):
        logw = beta * perf + np.where(pos <= 10, 0.0, math.log(0.3))
        logw = np.where(dnf, -np.inf, logw)
        g = rng.gumbel(size=perf.shape)
        winner = np.argmax(logw + g, axis=1)
        fl = np.zeros(perf.shape, bool)
        fl[np.arange(n), winner] = True
        return fl

    # ---- sprint ----------------------------------------------------
    comp, ccomp = {}, {}
    pos, dnf_d, fl_d = {"quali": q_pos}, {}, {}
    if is_sprint:
        s_pos, s_dnf, s_perf, s_ot = race_like("sprint", sq_pos, p.sprint_dnf_factor, 1.2, 0.5)
        s_fl = fastest_lap(s_pos, s_dnf, s_perf, p.fl_beta * 0.8)
        pos["sq"], pos["sprint"] = sq_pos, s_pos
        dnf_d["sprint"], fl_d["sprint"] = s_dnf, s_fl
        for k, v in R.score_race_like("sprint", sq_pos, s_pos, s_dnf, s_ot, s_fl).items():
            comp[("sprint", k)] = v
    else:
        z = np.zeros((n, nD), bool)
        dnf_d["sprint"], fl_d["sprint"] = z, z
        pos["sprint"] = np.zeros((n, nD), int)

    # ---- race grid -------------------------------------------------
    grid = q_pos.copy()
    if grid_penalties:
        key = grid.astype(float)
        for d, places in grid_penalties.items():
            if d in drivers:
                key[:, drivers.index(d)] += places + 0.5
        grid = _rank_positions(-key)

    r_pos, r_dnf, r_perf, r_ot = race_like("race", grid, 1.0, 1.0, 1.0)
    r_fl = fastest_lap(r_pos, r_dnf, r_perf, p.fl_beta)
    pos["race"] = r_pos
    dnf_d["race"], fl_d["race"] = r_dnf, r_fl

    for k, v in R.score_race_like("race", grid, r_pos, r_dnf, r_ot, r_fl).items():
        comp[("race", k)] = v
    qc = {"finish": R.score_qualifying(q_pos, q_nc)}
    comp[("quali", "finish")] = qc["finish"]

    # ---- driver of the day (heuristic) -----------------------------
    dotd = np.zeros((n, nD))
    if p.dotd:
        gain = (grid - r_pos).astype(float)
        s = (0.25 * np.clip(gain, -2, 10) + 1.0 * (r_pos == 1) + 0.5 * (r_pos <= 3)
             + 0.05 * np.minimum(r_ot, 10))
        s = np.where(r_dnf, -4.0, s)
        pr = np.exp(s - s.max(1, keepdims=True))
        pr /= pr.sum(1, keepdims=True)
        cum = pr.cumsum(1)
        win = (rng.random((n, 1)) > cum).sum(1).clip(0, nD - 1)
        dotd[np.arange(n), win] = R.DRIVER_OF_THE_DAY

    # ---- pit stops (constructor) -----------------------------------
    stops_mean = prof["stops"]
    stops = 1 + rng.binomial(2, np.clip((stops_mean - 1.0) / 2.0, 0.02, 0.98), size=(n, nD))
    stops = np.where(r_dnf, np.floor(stops * rng.random((n, nD))).astype(int), stops)
    mu = np.array([model.pit.mu_for(model.field[d]) for d in drivers])
    pit_t = np.full((n, nD, 3), np.nan)
    for s_i in range(3):
        z = rng.standard_normal((n, nD))
        sig = np.where(z < 0, PitModel.SIGMA_LO, PitModel.SIGMA_HI)
        t = np.exp(mu[None, :] + sig * z)
        slow = rng.random((n, nD)) < PitModel.P_SLOW
        t = np.where(slow, t + rng.uniform(0.8, 5.0, size=(n, nD)), t)
        pit_t[:, :, s_i] = np.where(stops > s_i, t, np.nan)
    stop_pts = np.nansum(R.pit_stop_points(pit_t), axis=2)           # (n, nD)
    team_min = np.full((n, nT), np.inf)
    for i in range(nD):
        m = np.nanmin(np.where(np.isnan(pit_t[:, i, :]), np.inf, pit_t[:, i, :]), axis=1)
        team_min[:, team_of[i]] = np.minimum(team_min[:, team_of[i]], m)
    fastest_team = np.argmin(team_min, axis=1)

    # ---- constructor scoring ---------------------------------------
    def team_sum(arr):
        out = np.zeros((n, nT))
        for i in range(nD):
            out[:, team_of[i]] += arr[:, i]
        return out

    q3 = np.zeros((n, nT), int)
    q2 = np.zeros((n, nT), int)
    for i in range(nD):
        eff = np.where(q_nc[:, i], 10 ** 6, q_pos[:, i])
        q3[:, team_of[i]] += eff <= R.Q3_CUTOFF
        q2[:, team_of[i]] += eff <= R.Q2_CUTOFF
    B = R.CONSTRUCTOR_QUALI_Q2_Q3
    bonus = np.where(q3 >= 2, B["both_q3"], np.where(q3 == 1, B["one_q3"],
             np.where(q2 >= 2, B["both_q2"], np.where(q2 == 1, B["one_q2"], B["neither_q2"])))).astype(float)
    ccomp[("quali", "finish")] = team_sum(comp[("quali", "finish")])
    ccomp[("quali", "progression")] = bonus
    for (e, k), v in list(comp.items()):
        if e != "quali":
            ccomp[(e, k)] = team_sum(v)
    pit_pts = team_sum(stop_pts)
    pit_pts[np.arange(n), fastest_team] += R.PIT_STOP_FASTEST_BONUS
    ccomp[("race", "pit")] = pit_pts

    f32 = lambda a: a.astype(np.float32)  # noqa: E731
    return WeekendForecast(
        drivers=drivers, teams=teams, team_of=list(team_of), race=race_info, n=n,
        sprint=bool(is_sprint), pos=pos, dnf=dnf_d, fl=fl_d,
        comp={k: f32(v) for k, v in comp.items()}, dotd=f32(dotd),
        ccomp={k: f32(v) for k, v in ccomp.items()},
        meta={"q_cal": [model.q_cal.c, model.q_cal.rho], "r_cal": [model.r_cal.c, model.r_cal.rho],
              "kappa": kappa, "track": prof, "overtake_params": list(model.ot),
              "calibration_weekends": [model.q_cal.n_weekends, model.r_cal.n_weekends]},
    )


# ──────────────────────────────────────────────────────────────
# High-level API
# ──────────────────────────────────────────────────────────────

class WeekendForecaster:
    """Loads results once, then forecasts any round of the target season."""

    def __init__(self, results=None, params=None, year=2026, verbose=True):
        self.res = results if results is not None else load_results(verbose=verbose)
        self.params = params or ModelParams()
        self.year = year
        self._calib = None
        self._cache = {}

    def _history(self, round_num):
        """Results strictly before (year, round_num); returns (Results, now_t)."""
        if self.res.empty:
            return self.res, 0.0
        wk = self.res.weekends
        past = wk[(wk["year"] < self.year) | ((wk["year"] == self.year) & (wk["round"] < round_num))]
        if past.empty:
            return Results(pd.DataFrame(), pd.DataFrame(), pd.DataFrame({"t": []})), 0.0
        keep = set(past["t"])
        s = self.res.sessions[self.res.sessions["t"].isin(keep)]
        pits = self.res.pits[self.res.pits["t"].isin(keep)] if not self.res.pits.empty else self.res.pits
        last = past.iloc[-1]
        gap = max(1, round_num - last["round"]) if last["year"] == self.year else 1
        return Results(s, pits, past), float(past["t"].max() + gap)

    def forecast(self, round_num, n=10000, seed=7, known=None, exclude=(), grid_penalties=None,
                 field_=None):
        race = get_race_info(round_num)
        if race is None:
            raise ValueError(f"Unknown round {round_num}")
        hist, now_t = self._history(round_num)
        field_ = field_ or {d: t for d, t in GRID_2026.items() if d not in exclude}
        key = (round_num, tuple(sorted(field_.items())), len(hist.weekends))
        model = self._cache.get(key)
        if model is None:
            model = fit_model(hist, field_, self.year, now_t, self.params, self._calib)
            self._calib = (model.q_cal, model.r_cal)
            self._cache[key] = model
        return simulate_weekend(model, race, n=n, seed=seed, known=known,
                                grid_penalties=grid_penalties)

    def ratings(self, round_num):
        hist, now_t = self._history(round_num)
        field_ = dict(GRID_2026)
        return fit_model(hist, field_, self.year, now_t, self.params, self._calib).summary()


# ──────────────────────────────────────────────────────────────
# Backtest
# ──────────────────────────────────────────────────────────────

def _spearman(a, b):
    a, b = pd.Series(a), pd.Series(b)
    if len(a) < 3 or a.nunique() < 2 or b.nunique() < 2:
        return np.nan
    return float(a.rank().corr(b.rank()))


def backtest(res, years=(2025,), n_sims=3000, params=None, seed=11, verbose=True,
             recalibrate_every=6, max_weekends=None):
    """
    Walk-forward evaluation.  For each weekend of `years` the model is fitted on
    strictly earlier weekends only, simulated, and compared with what happened.
    Driver of the Day and pit-stop points are excluded from the comparison
    (neither is observable in the data).
    """
    params = replace(params or ModelParams(), dotd=False)
    wk = res.weekends[res.weekends["year"].isin(years)]
    if max_weekends:
        wk = wk.tail(max_weekends)
    rows, calib, since = [], None, 0
    for _, w in wk.iterrows():
        t, year, rnd = int(w["t"]), int(w["year"]), int(w["round"])
        past = res.weekends[res.weekends["t"] < t]
        if len(past) < params.min_calib_history:
            continue
        keep = set(past["t"])
        hist = Results(res.sessions[res.sessions["t"].isin(keep)],
                       res.pits[res.pits["t"].isin(keep)] if not res.pits.empty else res.pits, past)
        sess = res.sessions[res.sessions["t"] == t]
        field_ = dict(zip(sess["driver"], sess["team"]))
        race = {"round": rnd, "location": w["location"], "sprint": bool(w["sprint"]), "name": w["location"]}
        if since % recalibrate_every == 0:
            calib = None
        model = fit_model(hist, field_, year, float(t), params, calib)
        calib, since = (model.q_cal, model.r_cal), since + 1
        fc = simulate_weekend(model, race, n=n_sims, seed=seed + t)
        actual_d, actual_c = score_weekend_actual(res, t)
        if actual_d.empty:
            continue

        pred_tot = fc.total_points(include_dotd=False)
        sim_tot = dict(zip(fc.drivers, pred_tot.mean(0)))
        sims = {d: pred_tot[:, i] for i, d in enumerate(fc.drivers)}
        # naive baseline: mean of the driver's previous 3 weekend totals
        base = {}
        for d in fc.drivers:
            prev = []
            for tt in sorted(keep, reverse=True)[:3]:
                ad, _ = score_weekend_actual(res, tt)
                if not ad.empty and d in set(ad["driver"]):
                    prev.append(float(ad.loc[ad["driver"] == d, "total"].iloc[0]))
            base[d] = np.mean(prev) if prev else np.nan

        a = actual_d.set_index("driver")
        common = [d for d in fc.drivers if d in a.index]
        act = np.array([a.loc[d, "total"] for d in common])
        pr = np.array([sim_tot[d] for d in common])
        bs = np.array([base[d] for d in common], dtype=float)
        bs = np.where(np.isnan(bs), np.nanmean(bs) if np.isfinite(np.nanmean(bs)) else act.mean(), bs)
        cover = np.mean([np.percentile(sims[d], 10) <= a.loc[d, "total"] <= np.percentile(sims[d], 90)
                         for d in common])
        # position metrics
        qa = a["quali_pos"] if "quali_pos" in a else None
        ra = a["race_pos"] if "race_pos" in a else None
        exp_q = pd.Series(fc.pos["quali"].mean(0), index=fc.drivers)
        exp_r = pd.Series(fc.pos["race"].mean(0), index=fc.drivers)
        q_rho = _spearman(exp_q[common], qa[common]) if qa is not None else np.nan
        dnf_act = np.array([float(a.loc[d, "race_dnf"]) if "race_dnf" in a else 0.0 for d in common])
        dnf_p = np.array([fc.dnf["race"][:, fc.drivers.index(d)].mean() for d in common])
        pole_p = pd.Series((fc.pos["quali"] == 1).mean(0), index=fc.drivers)
        win_p = pd.Series(((fc.pos["race"] == 1) & ~fc.dnf["race"]).mean(0), index=fc.drivers)
        act_pole = qa[common].idxmin() if qa is not None else None
        act_win = ra[common].replace(0, np.nan).idxmin() if ra is not None and ra[common].replace(0, np.nan).notna().any() else None
        rows.append({
            "year": year, "round": rnd, "location": w["location"],
            "mae_sim": float(np.abs(act - pr).mean()), "mae_naive": float(np.abs(act - bs).mean()),
            "corr_sim": _spearman(pr, act), "corr_naive": _spearman(bs, act),
            "quali_rank_corr": q_rho,
            "coverage80": float(cover),
            "dnf_brier": float(((dnf_p - dnf_act) ** 2).mean()),
            "dnf_brier_base": float(((0.07 - dnf_act) ** 2).mean()),
            "pole_p_of_actual": float(pole_p.get(act_pole, np.nan)) if act_pole else np.nan,
            "win_p_of_actual": float(win_p.get(act_win, np.nan)) if act_win else np.nan,
            "pole_hit": bool(pole_p.idxmax() == act_pole) if act_pole else np.nan,
            "win_hit": bool(win_p.idxmax() == act_win) if act_win else np.nan,
        })
        if verbose:
            r = rows[-1]
            print(f"  {year} R{rnd:<2d} {w['location']:<18s} MAE sim {r['mae_sim']:5.2f} vs naive "
                  f"{r['mae_naive']:5.2f} | rank-corr {r['corr_sim']:5.2f} | 80%-cov {r['coverage80']:.2f}")
    df = pd.DataFrame(rows)
    if verbose and not df.empty:
        print("\n  ── Backtest summary ──")
        print(f"  weekends evaluated        : {len(df)}")
        print(f"  driver-total MAE          : sim {df.mae_sim.mean():.2f}  | naive(last-3) {df.mae_naive.mean():.2f}")
        print(f"  rank correlation vs actual: sim {df.corr_sim.mean():.3f} | naive {df.corr_naive.mean():.3f}")
        print(f"  quali rank correlation    : {df.quali_rank_corr.mean():.3f}")
        print(f"  80% interval coverage     : {df.coverage80.mean():.2f}   (ideal 0.80)")
        print(f"  DNF Brier                 : {df.dnf_brier.mean():.4f}  vs flat-7% {df.dnf_brier_base.mean():.4f}")
        print(f"  pole / winner top-pick hit: {df.pole_hit.dropna().mean():.2f} / {df.win_hit.dropna().mean():.2f}")
    return df


def tune(res, years=(2025,), grid=None, n_sims=1500, max_weekends=10):
    """
    Coarse grid-search of the main hyper-parameters on a short walk-forward backtest.
    half_life: how fast old sessions fade; team_ridge_new: how long the pre-season
    prior is trusted at the start of a regulation era.
    """
    from itertools import product
    grid = grid or {"half_life": [5.0, 10.0, 20.0], "team_ridge_new": [1.0, 2.0, 4.0]}
    keys = list(grid)
    best = None
    for combo in product(*(grid[k] for k in keys)):
        pr = ModelParams(**dict(zip(keys, combo)))
        df = backtest(res, years, n_sims, pr, verbose=False, max_weekends=max_weekends)
        if df.empty:
            continue
        score = df.mae_sim.mean()
        print("  " + "  ".join(f"{k}={v}" for k, v in zip(keys, combo))
              + f"  MAE={score:.3f}  rank-corr={df.corr_sim.mean():.3f}")
        if best is None or score < best[0]:
            best = (score, pr)
    return best


# ──────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────

def format_forecast(fc, top_n=None):
    race = fc.race
    out = []
    out.append("=" * 78)
    out.append(f"  R{race.get('round', '?')} {race.get('name', '')}  ({race.get('location', '')})"
               f"   {'SPRINT WEEKEND' if fc.sprint else ''}   [{fc.n:,} simulations]")
    out.append("=" * 78)
    dt = fc.driver_table()
    if top_n:
        dt = dt.head(top_n)
    cols = ["quali", "sprint", "race"] if fc.sprint else ["quali", "race"]
    out.append("\n  DRIVERS  (mean points by event; 10-90% range; probabilities)")
    head = f"  {'driver':<12}{'team':<16}" + "".join(f"{c.title():>7}" for c in cols) + \
        f"{'TOTAL':>8}{'  10-90%':>12}{'Pole':>6}{'Win':>6}{'Pod':>6}{'Pts':>6}{'DNF':>6}"
    out.append(head)
    for _, r in dt.iterrows():
        out.append(f"  {r['driver']:<12}{r['team'][:15]:<16}" + "".join(f"{r[c]:7.1f}" for c in cols)
                   + f"{r['total']:8.1f}{f'{r.p10:.0f}-{r.p90:.0f}':>12}"
                   + f"{r.P_pole:6.0%}{r.P_win:6.0%}{r.P_podium:6.0%}{r.P_points:6.0%}{r.P_dnf:6.0%}")
    ct = fc.constructor_table()
    out.append("\n  CONSTRUCTORS")
    out.append(f"  {'team':<18}" + "".join(f"{c.title():>7}" for c in cols)
               + f"{'TOTAL':>8}{'  10-90%':>12}{'Pit':>6}{'2xQ3':>6}")
    for _, r in ct.iterrows():
        out.append(f"  {r['team']:<18}" + "".join(f"{r[c]:7.1f}" for c in cols)
                   + f"{r['total']:8.1f}{f'{r.p10:.0f}-{r.p90:.0f}':>12}{r.pit_pts:6.1f}{r.P_both_q3:6.0%}")
    return "\n".join(out)


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def _cli():
    ap = argparse.ArgumentParser(description="F1 Fantasy weekend simulator")
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("forecast", help="forecast every event of a weekend")
    f.add_argument("--round", type=int, required=True)
    f.add_argument("--sims", type=int, default=20000)
    f.add_argument("--quali", help="comma-separated qualifying order (best first) to condition on")
    f.add_argument("--sprint-quali", help="comma-separated sprint qualifying order")
    f.add_argument("--exclude", help="comma-separated drivers not racing")
    f.add_argument("--penalty", action="append", default=[], help="Driver=places, e.g. Norris=3")
    f.add_argument("--driver", help="print the full component breakdown for one driver/team")
    f.add_argument("--json", help="write the forecast to this JSON file")

    b = sub.add_parser("backtest", help="walk-forward evaluation on past weekends")
    b.add_argument("--years", type=int, nargs="+", default=[2025])
    b.add_argument("--sims", type=int, default=3000)
    b.add_argument("--last", type=int, help="only the most recent N weekends")

    t = sub.add_parser("tune", help="tune half-life / shrinkage on a short backtest")
    t.add_argument("--years", type=int, nargs="+", default=[2025])

    r = sub.add_parser("ratings", help="show fitted driver ratings")
    r.add_argument("--round", type=int, default=1)

    sub.add_parser("demo", help="run end-to-end on synthetic data")

    a = ap.parse_args()

    if a.cmd == "demo":
        from f1_synthetic import synthetic_results
        res = synthetic_results(seed=3, years=(2024, 2025, 2026), rounds_2026=6)
        print(f"  (synthetic data: {len(res.weekends)} weekends)\n")
        fcster = WeekendForecaster(res, verbose=False)
        print(format_forecast(fcster.forecast(7, n=8000), top_n=12))
        print()
        backtest(res, years=(2026,), n_sims=1500)
        return

    res = load_results()
    if res.empty and a.cmd != "forecast":
        raise SystemExit("No race data found (need raw_historical_data.json or "
                         "f1_driver_training_data.csv). Run f1_trainer.py first, or try `demo`.")
    if a.cmd == "forecast":
        fcster = WeekendForecaster(res)
        known = {}
        if a.quali:
            known["quali"] = [s.strip() for s in a.quali.split(",")]
        if a.sprint_quali:
            known["sprint_quali"] = [s.strip() for s in a.sprint_quali.split(",")]
        pen = {k: float(v) for k, v in (p.split("=") for p in a.penalty)}
        fc = fcster.forecast(a.round, n=a.sims, known=known,
                             exclude=tuple(s.strip() for s in a.exclude.split(",")) if a.exclude else (),
                             grid_penalties=pen)
        if res.empty:
            print("  ⚠ No race history found - using the pre-season price prior only.\n")
        print(format_forecast(fc))
        if a.driver:
            print(f"\n  Breakdown for {a.driver}: {json.dumps(fc.breakdown(a.driver), indent=2)}")
        if a.json:
            with open(a.json, "w") as fh:
                json.dump(fc.to_dict(), fh, indent=1)
            print(f"\n  Saved {a.json}")
    elif a.cmd == "backtest":
        backtest(res, tuple(a.years), a.sims, max_weekends=a.last)
    elif a.cmd == "tune":
        best = tune(res, tuple(a.years))
        if best:
            print(f"\n  Best: half_life={best[1].half_life}, team_ridge_new={best[1].team_ridge_new}"
                  f"  (MAE {best[0]:.3f})  - set these in ModelParams to use them")
    elif a.cmd == "ratings":
        print(WeekendForecaster(res).ratings(a.round).round(3).to_string(index=False))


if __name__ == "__main__":
    _cli()
