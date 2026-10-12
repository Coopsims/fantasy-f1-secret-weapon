#!/usr/bin/env python3
"""
Price reconstruction (f1_prices.py)
===================================
Rebuilds the weekend-by-weekend fantasy prices from the points every driver /
constructor scored, by replaying the game's price-change rule, and fits the parts
of that rule we do not know to whatever real prices are available.

The rule as modelled (2025-season guides; the 2026 game may differ, which is why it is
fitted to real prices whenever there are any):

  * two price tiers split at $19m (FanAmp 2025 guide text; its graphic says $18.5m):
        A tier (price >= tier_cut):  GREAT +0.3  GOOD +0.1  POOR -0.1  TERRIBLE -0.3
        B tier (price <  tier_cut):  GREAT +0.6  GOOD +0.2  POOR -0.2  TERRIBLE -0.6
  * performance = points per $m (PPM) of each race, i.e. that race's points divided by
    the price paid for it, averaged over the last 3 races; ratings at PPM cut-offs
    0.6 / 0.9 / 1.2 (terrible < 0.6 <= poor < 0.9 <= good < 1.2 <= great).
  * early season the game pads the 3-race window with zero-point races and lowers the
    cut-offs to match (Australia: 0.2 / 0.3 / 0.4, China: 0.4 / 0.6 / 0.8, then
    0.6 / 0.9 / 1.2 from race 3) - which is exactly the
    same as averaging over the races run so far against 0.6 / 0.9 / 1.2, so that is
    what is implemented.  Worked example from the guide: $23.0m driver scoring 10 in
    Australia -> PPM 0.43 -> terrible -> -$0.3m (tested).
  * prices never fall below `floor` = $3.0m and never rise above `cap` = $34.0m, and a driver who does not take
    part in a weekend keeps his price.  These three come from the game's official 2026 prices (fantasy feed):
    assets sat at $3.0m 30 times and never went lower; Mercedes went 33.8 -> 34.0 (+0.2 where the rule says +0.3)
    and then stayed (only two observations, so the cap is "observed", not certain); Hadjar stayed at $14.5m while
    absent (rounds 14-16) and so did the Racing Bulls Lawson asset while he drove for Red Bull.
  * tier split: the A tier starts at $18.5m (Alpine moved in A-tier steps from $18.5m; matches the FanAmp graphic).

Accuracy against the official 2026 prices (experiments/price_rule_check.py): the guide's default rule driven by the
game's official points reproduces 94% of week-to-week moves exactly and ends the season $0.19m (MAE) from the real
prices; driven by our own reconstructed points it reproduces 77%.  FITTING the cut-offs to only two real weekends made
it much worse (drift $1.41m MAE by Singapore), so `reconstruct` only fits with >= `min_fit_segments` (6) transitions.

What makes a reconstruction trustworthy is real prices to anchor on.  Known prices
(any round, any subset of assets) are used as checkpoints: the rule is fitted to
reproduce them, and the reconstruction snaps back to them wherever they exist, so
errors cannot accumulate across a checkpoint.  `fit_rule` reports how well the
fitted rule reproduces the checkpoints it was given (and `holdout_check` how well it
predicts ones it was not given) - read those numbers before trusting the result.
"""

from dataclasses import dataclass, replace
from itertools import product

import numpy as np
import pandas as pd

LEVELS = ("TERRIBLE", "POOR", "GOOD", "GREAT")
STEP = {"A": np.array([-0.3, -0.1, 0.1, 0.3]), "B": np.array([-0.6, -0.2, 0.2, 0.6])}


@dataclass(frozen=True)
class PriceRule:
    tier_cut: float = 18.5            # A tier from $18.5m (official 2026 prices; FanAmp graphic 18.5, its text 19)
    cuts_a: tuple = (0.6, 0.9, 1.2)   # PPM cut-offs poor / good / great, A tier (2025 guide)
    cuts_b: tuple = (0.6, 0.9, 1.2)   # same for B tier (the guides give one set for both)
    window: int = 3
    floor: float = 3.0                # official 2026 prices (lowest ever, held 30 times)
    cap: float = 34.0                 # official 2026 prices (Mercedes clipped at 34.0; only two observations)


def level_index(ppm, cuts):
    """0 TERRIBLE, 1 POOR, 2 GOOD, 3 GREAT."""
    return np.searchsorted(np.asarray(cuts), ppm, side="right")


def step_prices(price, ppm_window, rule, active=None):
    """
    Prices after one raced weekend.
    price      : (E,) price held into the weekend just raced (the one that moves)
    ppm_window : (k, E) points-per-$m of the last k <= window races, the latest last
    active     : (E,) bool - False for an asset that did not take part (its price does not move)
    """
    ppm = np.asarray(ppm_window, float).mean(0)
    a = price >= rule.tier_cut
    lv = np.where(a, level_index(ppm, rule.cuts_a), level_index(ppm, rule.cuts_b))
    delta = np.where(a, STEP["A"][lv], STEP["B"][lv])
    if active is not None:
        delta = np.where(np.asarray(active, bool), delta, 0.0)
    return np.minimum(rule.cap, np.maximum(rule.floor, np.round(price + delta, 1)))


def _ppm(pts, P, j):
    """PPM of race j (1-based): its points over the price paid for it (P[j-1])."""
    return np.asarray(pts, float)[j - 1] / np.maximum(P[j - 1], 0.1)


def simulate(pts, start, rule, anchors=None, upto=None, history=None, active=None):
    """
    prices[j] = prices after j raced weekends (j = 0..R).  pts: (R, E).
    anchors: {j: vector, NaN = unknown} - known prices replace simulated ones where present.
    active : optional (R, E) bool - who took part in each weekend (non-participants keep their price).
    """
    pts = np.asarray(pts, float)
    R, E = pts.shape
    anchors = anchors or {}
    P = np.empty((R + 1, E))
    P[0] = _merge(start, anchors.get(0))
    ppm = np.zeros((R + 1, E))
    for j in range(1, R + 1):
        ppm[j] = _ppm(pts, P, j)
        sim = step_prices(P[j - 1], ppm[max(1, j - rule.window + 1):j + 1], rule,
                          None if active is None else active[j - 1])
        P[j] = _merge(sim, anchors.get(j))
    return P


def _merge(base, known):
    base = np.asarray(base, float)
    if known is None:
        return base.copy()
    known = np.asarray(known, float)
    return np.where(np.isnan(known), base, known)


# ──────────────────────────────────────────────────────────────
# Checkpoints and fitting
# ──────────────────────────────────────────────────────────────

def checkpoints_from_csv(df, raced_rounds, names):
    """
    Price-history CSV rows (round, name, price) -> {j: vector}, j = raced weekends completed.

    A row labelled `round` k means "prices after round k".  With cancelled rounds the
    calendar round and the number of races run differ, so k is mapped to the count of
    raced rounds <= k; if several labels map to the same j the latest label wins.
    """
    out = {}
    raced = np.asarray(sorted(raced_rounds))
    for k, g in sorted(df.groupby("round"), key=lambda kv: kv[0]):
        j = int((raced <= int(k)).sum())
        s = g.set_index("name")["price"]
        vec = np.array([float(s[n]) if n in s.index else np.nan for n in names])
        prev = out.get(j)
        out[j] = vec if prev is None else np.where(np.isnan(vec), prev, vec)
    return out


def _segments(anchors):
    js = sorted(anchors)
    return list(zip(js[:-1], js[1:]))


def segment_errors(pts, anchors, rule, path=None, active=None):
    """
    For each consecutive pair of checkpoints (a, b): start from the real prices at a,
    replay the rule to b, compare with the real prices at b.  Earlier races needed for the
    3-race window come from the anchored path.  Returns rows of (a, b, n_compared, mae, exact_share).
    """
    pts = np.asarray(pts, float)
    R = len(pts)
    if path is None:
        path = simulate(pts, anchors.get(0), rule, anchors, active=active)
    rows = []
    for a, b in _segments(anchors):
        if b > R:
            continue
        P = path.copy()                 # path[a] = real prices at a, gaps filled by the rule
        for j in range(a + 1, b + 1):
            win = [_ppm(pts, P, i) for i in range(max(1, j - rule.window + 1), j + 1)]
            P[j] = step_prices(P[j - 1], win, rule, None if active is None else active[j - 1])
        tgt = anchors[b]
        m = ~np.isnan(tgt)
        if not m.any():
            continue
        err = np.abs(P[b][m] - tgt[m])
        rows.append((a, b, int(m.sum()), float(err.mean()), float((err < 0.05).mean())))
    return rows


def _loss(pts, anchors, rule, active=None):
    rows = segment_errors(pts, anchors, rule, active=active)
    if not rows:
        return np.inf
    n = sum(r[2] for r in rows)
    return sum(r[2] * r[3] for r in rows) / n


def fit_rule(pts, anchors, base=None, tier_cuts=(18.0, 18.5, 19.0, 20.0),
             scales=np.round(np.arange(0.5, 1.81, 0.05), 3), refine=True, active=None):
    """
    Fit the rule to the checkpoints.  Stage 1: tier cut-off x a common scale on the guide's
    cut-offs (0.6, 0.9, 1.2).  Stage 2 (refine): nudge each cut-off of each tier on its own.
    Returns (rule, loss_mae); loss is inf when there is nothing to fit against.
    """
    base = base or PriceRule()
    if not _segments(anchors):
        return base, np.inf
    g = np.array((0.6, 0.9, 1.2))
    best = (np.inf, base)
    for tc, sc in product(tier_cuts, scales):
        cuts = tuple(np.round(g * sc, 3))
        r = replace(base, tier_cut=tc, cuts_a=cuts, cuts_b=cuts)
        l = _loss(pts, anchors, r, active)
        if l < best[0] - 1e-12:
            best = (l, r)
    if refine:
        for _ in range(2):
            for which, k in product(("cuts_a", "cuts_b"), range(3)):
                cur = list(getattr(best[1], which))
                for d in np.arange(-0.3, 0.301, 0.025):
                    c = cur.copy()
                    c[k] = round(cur[k] + d, 3)
                    if not (c[0] < c[1] < c[2]):
                        continue
                    r = replace(best[1], **{which: tuple(c)})
                    l = _loss(pts, anchors, r, active)
                    if l < best[0] - 1e-12:
                        best = (l, r)
    return best[1], best[0]


def holdout_check(pts, anchors, **fit_kw):
    """
    Honest accuracy estimate: for every checkpoint after the first, fit on the others and
    predict it.  Returns DataFrame(j, mae, exact_share) - empty with fewer than 3 checkpoints.
    """
    js = sorted(anchors)
    rows = []
    if len(js) < 3:
        return pd.DataFrame(columns=["j", "mae", "exact_share"])
    for hold in js[1:]:
        train = {j: v for j, v in anchors.items() if j != hold}
        rule, _ = fit_rule(pts, train, **fit_kw)
        prev = max(j for j in train if j < hold)
        rows_ = segment_errors(pts, {prev: train[prev], hold: anchors[hold]}, rule,
                               path=simulate(pts, train[min(train)], rule, train))
        if rows_:
            rows.append({"j": hold, "mae": rows_[0][3], "exact_share": rows_[0][4]})
    return pd.DataFrame(rows)


def reconstruct(pts, start, anchors=None, rule=None, fit=True, verbose=True, active=None, min_fit_segments=6):
    """
    Full reconstruction.  Returns (P, info): P[j] = prices after j raced weekends, so the
    price paid for raced weekend i (0-based) is P[i].  info = {rule, fit_mae, segments, n_estimated}.
    The rule's cut-offs are only fitted with at least `min_fit_segments` transitions between real prices;
    with fewer, the guide's rule is used (a 2-transition fit drifted 7x further than the default in 2026).
    """
    anchors = dict(anchors or {})
    anchors.setdefault(0, np.asarray(start, float))
    anchors[0] = _merge(start, anchors[0])
    info = {"rule": rule or PriceRule(), "fit_mae": None, "segments": [], "fitted": False}
    if rule is None and fit and len(_segments(anchors)) >= max(1, min_fit_segments):
        info["rule"], info["fit_mae"] = fit_rule(pts, anchors, active=active)
        info["fitted"] = True
    info["segments"] = segment_errors(pts, anchors, info["rule"], active=active)
    P = simulate(pts, anchors[0], info["rule"], anchors, active=active)
    R = len(pts)
    info["n_estimated"] = sum(1 for j in range(1, R + 1) if j not in anchors or np.isnan(anchors[j]).any())
    if verbose:
        r = info["rule"]
        src = ("fitted to your real prices" if info["fitted"] else
               "guide defaults (too few real prices to fit)" if anchors and len(anchors) > 1 else "guide defaults (no real prices)")
        print(f"  Price rule ({src}): A tier >= ${r.tier_cut}m, pts/$m cut-offs A {r.cuts_a} B {r.cuts_b}")
        for a, b, n, mae, ex in info["segments"]:
            print(f"    real prices after {a} -> {b} races: {n} assets, MAE ${mae:.2f}m, exact {ex:.0%}")
    return P, info
