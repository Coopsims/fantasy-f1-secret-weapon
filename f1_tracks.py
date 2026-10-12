#!/usr/bin/env python3
"""
Track groups (f1_tracks.py)
===========================
Describes every circuit from car telemetry (fastest PRACTICE laps before the team lock, fetch_track_telemetry.py)
plus web facts (data/circuits_web.csv), groups similar circuits, and gives a similarity matrix used for
"how has this driver / team done at tracks like this one" features.

Leakage rules: practice laps only (never qualifying or race); descriptors are standardised within each season (so the
faster 2026 cars do not look like a different kind of track); circuits that already existed are described from
2023-2025 practice only; circuits new in 2026 (Madrid, Kuala Lumpur) from their own 2026 practice, which happens before
the lock.  No finishing results are used to build the groups.
"""
import glob
import gzip
import json
import os

import numpy as np
import pandas as pd

from f1_calendar import canon_location

TELEMETRY_DIR = os.path.join("openf1_full", "telemetry")
WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "circuits_web.csv")
PHYS = ["v_mean", "v_max", "v_min", "full_throttle", "braking", "brake_zones_per_km", "slow_per_km", "medium_per_km",
        "fast_per_km", "gear_changes_per_km"]
WEBF = ["street", "turns_per_km", "length_km"]


def lap_descriptors(samples):
    d = pd.DataFrame(samples)
    d = d[d["speed"].notna()].copy()
    if len(d) < 50:
        return None
    d["_t"] = pd.to_datetime(d["date"], utc=True, format="ISO8601")
    d = d.sort_values("_t")                                   # the API does not return samples in time order
    t = d["_t"]
    dt = t.diff().dt.total_seconds().fillna(0).clip(0, 1.0).values
    v = d["speed"].astype(float).values
    T = dt.sum()
    dist = float((v / 3.6 * dt).sum() / 1000)
    if T < 30 or dist < 1.0:                                  # broken sample (no usable timing)
        return None
    sm = pd.Series(v).rolling(3, center=True, min_periods=1).mean().values
    mins = []
    for i in range(1, len(sm) - 1):                        # corners = local speed minima with >= 15 km/h prominence
        if sm[i] <= sm[i - 1] and sm[i] < sm[i + 1]:
            left, right = sm[max(0, i - 25):i].max(), sm[i + 1:i + 26].max()
            if min(left, right) - sm[i] >= 15:
                mins.append(sm[i])
    mins = np.array(mins)
    brake = d["brake"].astype(float).values > 0
    zones = int(((~brake[:-1]) & brake[1:]).sum())
    gears = d["n_gear"].astype(float).values
    return {"lap_km": dist, "v_mean": float(v.mean()), "v_max": float(v.max()), "v_min": float(v.min()),
            "full_throttle": float(dt[d["throttle"].astype(float).values >= 98].sum() / T),
            "braking": float(dt[brake].sum() / T), "brake_zones_per_km": zones / dist,
            "slow_per_km": float((mins < 130).sum() / dist), "medium_per_km": float(((mins >= 130) & (mins < 200)).sum() / dist),
            "fast_per_km": float((mins >= 200).sum() / dist), "gear_changes_per_km": float((np.diff(gears) != 0).sum() / dist)}


def weekend_descriptors(directory=TELEMETRY_DIR, min_coverage=0.95):
    """Per weekend: mean descriptors of its usable practice laps.  A lap is dropped when its integrated distance is
    under `min_coverage` of the official length (telemetry gaps; e.g. 2026 Shanghai, a 60 s 'lap' in 2026 Miami)."""
    length = pd.read_csv(WEB).set_index("location")["length_km"]
    rows = []
    for path in sorted(glob.glob(os.path.join(directory, "*.json.gz"))):
        e = json.load(gzip.open(path, "rt"))
        m = e["meeting"]
        L = length.get(canon_location(m["location"]), np.nan)
        laps = [lap_descriptors(l["samples"]) for l in e["laps"] if l["status"] == "ok"]
        laps = [x for x in laps if x and (np.isnan(L) or x["lap_km"] >= min_coverage * L)]
        if not laps:
            continue
        rows.append({"year": m["year"], "location": canon_location(m["location"]), "n_laps": len(laps),
                     **pd.DataFrame(laps).mean().to_dict()})
    return pd.DataFrame(rows)


def circuit_profiles(wd=None, through_year=2025):
    """One row per circuit: season-standardised telemetry descriptors (existing circuits: <= through_year) + web facts."""
    wd = weekend_descriptors() if wd is None else wd
    z = wd.copy()
    for c in PHYS:
        z[c] = z.groupby("year")[c].transform(lambda s: (s - s.mean()) / (s.std() or 1.0))
    seen_before = set(z[z["year"] <= through_year]["location"])
    use = z[(z["year"] <= through_year) | ~z["location"].isin(seen_before)]
    prof = use.groupby("location")[PHYS + ["lap_km"]].mean()
    web = pd.read_csv(WEB).set_index("location")
    prof = prof.join(web[["type", "length_km", "turns", "direction"]], how="left")
    prof["street"] = (prof["type"] == "street").astype(float)
    prof["turns_per_km"] = prof["turns"] / prof["length_km"]
    return prof


def _matrix(prof, web_weight=0.5):
    from sklearn.preprocessing import StandardScaler
    Xp = StandardScaler().fit_transform(prof[PHYS].values)
    Xw = StandardScaler().fit_transform(prof[WEBF].fillna(prof[WEBF].mean()).values) * web_weight
    return np.hstack([Xp, Xw])


def cluster(prof, ks=range(3, 8), web_weight=0.5, seed=0):
    """Ward clustering for each k; returns (labels for the best-silhouette k, table of silhouettes)."""
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    M = _matrix(prof, web_weight)
    sil = {}
    labels = {}
    for k in ks:
        lab = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(M)
        sil[k] = float(silhouette_score(M, lab))
        labels[k] = lab
    best = max(sil, key=sil.get)
    return pd.Series(labels[best], index=prof.index, name="track_group"), pd.Series(sil, name="silhouette"), best


def similarity(prof, web_weight=0.5, bandwidth=None):
    """Gaussian similarity between circuits on the standardised descriptor space (1 on the diagonal)."""
    M = _matrix(prof, web_weight)
    D = np.sqrt(((M[:, None, :] - M[None, :, :]) ** 2).sum(-1))
    bw = bandwidth or np.median(D[D > 0])
    S = np.exp(-(D / bw) ** 2)
    return pd.DataFrame(S, index=prof.index, columns=prof.index)
