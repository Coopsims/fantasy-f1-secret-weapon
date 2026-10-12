"""Data for the track-groups page: circuit map (practice telemetry) and who does well at each track type in 2026
(official fantasy points; weekends a driver did not race are left out).  Writes results/track_groups.json."""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, pandas as pd
import f1_tracks as T
import mlflow_experiments as X
from f1_calendar import RACE_CALENDAR_2026, GRID_2026

wd = T.weekend_descriptors()
prof = T.circuit_profiles(wd)
groups, sil, k = T.cluster(prof)
# raw (un-standardised) values for the hover, same selection rule as the profiles
seen = set(wd[wd.year <= 2025].location)
raw = wd[(wd.year <= 2025) | ~wd.location.isin(seen)].groupby("location")[["v_mean", "v_max", "v_min", "full_throttle", "braking", "slow_per_km", "fast_per_km"]].mean()
desc = prof.groupby(groups)[["v_mean", "braking", "slow_per_km", "fast_per_km", "full_throttle"]].mean()
order = desc["v_mean"].sort_values(ascending=False).index.tolist()       # fastest group first
names = {}
for g in order:
    n = int((groups == g).sum())
    names[g] = ("Monaco-type (tight street)" if n == 1 and "Monaco" in groups[groups == g].index else
                "Flowing, high-speed" if g == order[0] else "Stop-start, braking-heavy")
gid = {g: i for i, g in enumerate(order)}

res, truth, ctruth = X.get_data()
cal = {r["round"]: r["location"] for r in RACE_CALENDAR_2026}
op = pd.read_csv("fantasy_f1_official_points.csv")
op["location"] = op["round"].map(cal)
# a driver-weekend counts only if he raced it in his own 2026 car: stand-in weekends (Lawson for Red Bull, rounds 14-16)
# are scored by the game on a separate asset, so his model asset shows 0 there
raced = truth[truth.year == 2026][["round", "driver", "team"]].drop_duplicates()
raced = raced[raced["team"] == raced["driver"].map(GRID_2026)].assign(raced=True)[["round", "driver", "raced"]]
d = op[op.type == "driver"].merge(raced, left_on=["round", "name"], right_on=["round", "driver"], how="left")
d = d[d.raced.fillna(False)].drop(columns=["driver", "raced"])
c = op[op.type == "constructor"].copy()
for df in (d, c):
    df["group"] = df["location"].map(groups).map(gid)
rounds = op[["round", "location"]].drop_duplicates().assign(group=lambda x: x.location.map(groups).map(gid))
print("2026 weekends per track type:", rounds.groupby("group").size().rename(index={gid[g]: names[g] for g in gid}).to_dict())


def table(df):
    season = df.groupby("name")["points"].mean()
    out = []
    for (name, g), x in df.groupby(["name", "group"]):
        se = float(x.points.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else None
        out.append({"name": name, "group": int(g), "n": int(len(x)), "mean": float(x.points.mean()), "se": se,
                    "delta": float(x.points.mean() - season[name]), "season": float(season[name])})
    return out


def contrast(df, a=0, b=1):
    """Welch t for (mean at group a) - (mean at group b), per asset."""
    out = {}
    for name, x in df.groupby("name"):
        u, v = x[x.group == a].points, x[x.group == b].points
        if len(u) < 3 or len(v) < 3:
            continue
        diff = u.mean() - v.mean()
        se = np.sqrt(u.var(ddof=1) / len(u) + v.var(ddof=1) / len(v))
        out[name] = {"diff": float(diff), "se": float(se), "t": float(diff / se) if se > 0 else 0.0}
    return out


# ---- driver only: compared with the teammate in the same car, same weekend ----
d["team"] = d["name"].map(GRID_2026)
q = truth[truth.year == 2026][["round", "driver", "quali_pos"]].rename(columns={"driver": "name"})
d = d.merge(q, on=["round", "name"], how="left")
pairs = d.merge(d[["round", "team", "name", "points", "quali_pos"]], on=["round", "team"], suffixes=("", "_mate"))
pairs = pairs[pairs["name"] != pairs["name_mate"]].copy()
pairs["vs_mate"] = pairs["points"] - pairs["points_mate"]
pairs["q_vs_mate"] = (pairs["quali_pos_mate"] - pairs["quali_pos"]).where((pairs["quali_pos"] > 0) & (pairs["quali_pos_mate"] > 0))


def mate_table(df, col):
    out, con = [], {}
    for name, x in df.groupby("name"):
        x = x.dropna(subset=[col])
        overall = x[col].mean()
        for g, y in x.groupby("group"):
            out.append({"name": name, "group": int(g), "n": int(len(y)), "mean": float(y[col].mean()), "delta": float(y[col].mean() - overall),
                        "season": float(overall)})
        u, v = x[x.group == 0][col], x[x.group == 1][col]
        if len(u) >= 3 and len(v) >= 3:
            se = float(np.sqrt(u.var(ddof=1) / len(u) + v.var(ddof=1) / len(v)))
            con[name] = {"diff": float(u.mean() - v.mean()), "se": se, "t": float((u.mean() - v.mean()) / se) if se > 0 else 0.0}
    return out, con


mate_pts, mate_pts_con = mate_table(pairs, "vs_mate")
mate_q, mate_q_con = mate_table(pairs, "q_vs_mate")

payload = {
    "k": int(k), "silhouette": {int(a): float(b) for a, b in sil.items()},
    "groups": [{"id": gid[g], "name": names[g], "n_circuits": int((groups == g).sum())} for g in order],
    "circuits": [{"location": loc, "group": gid[groups[loc]], "x": float(prof.loc[loc, "v_mean"]), "y": float(prof.loc[loc, "slow_per_km"]),
                  "kmh": float(raw.loc[loc, "v_mean"]), "top": float(raw.loc[loc, "v_max"]), "min": float(raw.loc[loc, "v_min"]),
                  "throttle": float(raw.loc[loc, "full_throttle"]), "braking": float(raw.loc[loc, "braking"]),
                  "slow": float(raw.loc[loc, "slow_per_km"]), "fast": float(raw.loc[loc, "fast_per_km"]),
                  "street": bool(prof.loc[loc, "street"]), "length": float(prof.loc[loc, "length_km"]), "turns": int(prof.loc[loc, "turns"]),
                  "in_2026": bool(loc in set(rounds.location))} for loc in prof.index],
    "weekends_2026": {int(g): int(n) for g, n in rounds.groupby("group").size().items()},
    "drivers": table(d), "teams": table(c),
    "contrast_drivers": contrast(d), "contrast_teams": contrast(c),
    "mate_points": mate_pts, "mate_points_contrast": mate_pts_con, "mate_quali": mate_q, "mate_quali_contrast": mate_q_con,
    "team_of": GRID_2026,
}
os.makedirs("results", exist_ok=True)
json.dump(payload, open("results/track_groups.json", "w"), indent=1)
dt = pd.DataFrame(payload["drivers"])
for g in range(len(order)):
    x = dt[dt.group == g].sort_values("mean", ascending=False)
    print(f"\n{payload['groups'][g]['name']} ({payload['weekends_2026'].get(g, 0)} weekends in 2026): top by points/weekend:",
          [(r.name, round(r.mean, 1), r.n) for r in x.head(5).itertuples()])
    y = x[x.n >= 2].sort_values("delta", ascending=False)
    print("   most above own average:", [(r.name, round(r.delta, 1)) for r in y.head(4).itertuples()], "| below:", [(r.name, round(r.delta, 1)) for r in y.tail(3).itertuples()])

ct = pd.DataFrame(payload["contrast_drivers"]).T.sort_values("t")
print("\nflowing minus stop-start (points per weekend), |t| >= 2 means the difference is more than noise:")
print(ct.round(2).to_string())
print("teams:", pd.DataFrame(payload["contrast_teams"]).T.sort_values("t").round(2).to_string())

for lbl, con in (("points vs teammate", mate_pts_con), ("qualifying places vs teammate", mate_q_con)):
    t = pd.DataFrame(con).T.sort_values("t")
    print(f"\nflowing minus stop-start, {lbl} (car removed):"); print(t.round(2).to_string())
