import numpy as np
import pandas as pd

import f1_tracks as T
import mlflow_experiments as X
from experiments import zoo
from f1_synthetic import synthetic_results


def _lap(n=300, shuffle=False):
    t0 = pd.Timestamp("2026-03-06T01:30:00Z")
    rows = []
    for i in range(n):
        v = 200 + 80 * np.sin(i / 10.0)                     # ~ km/h, with corners every ~60 samples
        rows.append({"date": (t0 + pd.Timedelta(seconds=i * 0.27)).isoformat(), "speed": v, "throttle": 100 if v > 250 else 40,
                     "brake": 100 if (np.cos(i / 10.0) < -0.9) else 0, "n_gear": 7 if v > 240 else 4})
    if shuffle:
        rows = list(np.random.default_rng(0).permutation(rows))
    return rows


def test_lap_descriptors_do_not_depend_on_sample_order():
    a, b = T.lap_descriptors(_lap()), T.lap_descriptors(_lap(shuffle=True))     # the API returns samples unordered
    assert a is not None and all(np.isclose(a[k], b[k]) for k in a)
    assert abs(a["lap_km"] - 300 * 0.27 * 200 / 3.6 / 1000) < 1.0


def test_broken_lap_is_rejected():
    rows = _lap(80)
    for r in rows:
        r["date"] = rows[0]["date"]                          # no usable timing
    assert T.lap_descriptors(rows) is None


def test_panel_features_at_t_ignore_the_results_of_t_and_later():
    res = synthetic_results(seed=5, years=(2024, 2025), rounds_2026=None)
    truth, _ = X.build_truth(res)
    t0 = sorted(truth["t"].unique())[20]
    clean = zoo.build_panel(res, truth)
    tam = truth.copy()
    fut = tam["t"] >= t0
    tam.loc[fut, "total"] = tam.loc[fut, "total"].values[::-1] * 3 + 7
    tam.loc[fut, "race_pos"] = tam.loc[fut, "race_pos"].values[::-1]
    tam.loc[fut, "quali_pos"] = tam.loc[fut, "quali_pos"].values[::-1]
    dirty = zoo.build_panel(res, tam)
    cols = [c for c in zoo.FEATURES["form"] + zoo.FEATURES["team"] + ["d_track_mean"] if c in clean]
    a = clean[clean["t"] == t0].set_index("driver")[cols].sort_index()
    b = dirty[dirty["t"] == t0].set_index("driver")[cols].sort_index()
    assert np.allclose(a.values.astype(float), b.values.astype(float), equal_nan=True)


def test_walk_forward_trains_only_on_the_past():
    res = synthetic_results(seed=6, years=(2024, 2025), rounds_2026=None)
    truth, _ = X.build_truth(res)
    panel = zoo.build_panel(res, truth)
    ts = sorted(panel["t"].unique())[-3:]
    feats = [c for c in zoo.FEATURES["form"] if c in panel]
    a = zoo.walk_forward(panel, feats, "ridge", None, ts[:1], min_rows=50).drivers
    p2 = panel.copy(); p2.loc[p2["t"] >= ts[0], "total"] = 999.0             # poison the target weekend and later
    b = zoo.walk_forward(p2, feats, "ridge", None, ts[:1], min_rows=50).drivers
    assert np.allclose(a["pred"], b["pred"])


def test_team_affinity_never_sees_the_same_weekend():
    import pandas as pd
    from experiments.zoo import _sim_affinity
    locs = ["A", "B"] * 6
    rows = []
    for t in range(12):
        for d in ("x", "y"):                                     # two drivers of one team, same weekend
            rows.append({"t": t, "driver": d, "team": "T", "location": locs[t], "_tz": float(t)})
    df = pd.DataFrame(rows)
    S = pd.DataFrame([[1.0, 0.2], [0.2, 1.0]], index=["A", "B"], columns=["A", "B"])
    base = _sim_affinity(df, S, "team", "_tz")
    df2 = df.copy(); df2.loc[df2["t"] == 11, "_tz"] = 999.0       # change only the last weekend's outcome
    after = _sim_affinity(df2, S, "team", "_tz")
    last = df["t"].values == 11
    assert np.allclose(base[last], after[last], equal_nan=True)
