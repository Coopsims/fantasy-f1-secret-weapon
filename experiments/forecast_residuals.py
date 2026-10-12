"""Bank of 2026 forecast errors (official points - planning forecast), one row per gameday x asset, for the planners'
"how sure is this move" check (f1_planner.move_confidence).  Driver forecasts: mean of the interleaved run's tuned
versions of the deployed families (walk-forward; tuned on odd gamedays); constructors derived (drivers + recent bonus); both then
the planning blend (0.5 model + 0.3 simulator + 0.2 recent official form), as in f1_planner.forecast_remaining.
Replaying a whole gameday's errors keeps the correlation between a constructor and its drivers."""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, pandas as pd

if __name__ == "__main__":
    from experiments import s26
    from f1_calendar import GRID_2026
    import f1_planner as FP
    D = s26.prepare(os.environ["PANEL_PKL"], os.environ["S26_CACHE"])
    P = D["panel"]
    fs = json.load(open(FP.SPEC_FILE))
    pb, ba = fs["planning_blend"], fs["bonus_alpha"]
    preds = {}
    for path in os.environ.get("PREDS", "results/s26_interleaved-eval-fixed_best.pkl").split(","):
        preds |= pd.read_pickle(path)["preds"]
    fams = os.environ.get("FAMS") or ",".join(sp["family"] for sp in fs["specs"])          # default: the deployed families
    model = pd.concat([preds[f] for f in fams.split(",")]).groupby(["t", "driver"], as_index=False)["pred"].mean()
    d = model.merge(P[["t", "driver", "official", "sim", f"ew_off_{pb['ewma_alpha']}"]], on=["t", "driver"])
    d["sim"] = d["sim"].fillna(d["pred"])
    d["fc"] = pb["w_model"] * d["pred"] + pb["w_sim"] * d["sim"] + pb["w_ewma"] * d[f"ew_off_{pb['ewma_alpha']}"].fillna(d["sim"])
    # constructors: sum of the two drivers' model forecasts + bonus EWMA (earlier gamedays), then the same blend
    oc = D["official_c"].rename(columns={"points": "official"})
    dsum_off = P[P["year"] == 2026].assign(team26=lambda x: x["driver"].map(GRID_2026)).groupby(["t", "team26"])["official"].sum(min_count=1)
    extra = (oc.set_index(["t", "team"])["official"] - dsum_off.reindex(oc.set_index(["t", "team"]).index).values).rename("extra").reset_index()
    extra = extra.sort_values("t")
    extra["bonus"] = extra.groupby("team")["extra"].transform(lambda s: s.shift(1).ewm(alpha=ba, adjust=False, ignore_na=True).mean())
    oc = oc.sort_values("t")
    oc["ew"] = oc.groupby("team")["official"].transform(lambda s: s.shift(1).ewm(alpha=pb["ewma_alpha"], adjust=False, ignore_na=True).mean())
    c = (d.assign(team=d["driver"].map(GRID_2026)).groupby(["t", "team"], as_index=False)["pred"].sum()
         .merge(extra[["t", "team", "bonus"]], on=["t", "team"], how="left")
         .merge(D["sim_c"].rename(columns={"pred": "sim_c"}), on=["t", "team"]).merge(oc, on=["t", "team"]))
    c["model"] = c["pred"] + c["bonus"].fillna(0.0)
    c["fc"] = pb["w_model"] * c["model"] + pb["w_sim"] * c["sim_c"] + pb["w_ewma"] * c["ew"].fillna(c["sim_c"])
    rows = pd.concat([d.assign(asset=d["driver"], type="driver")[["t", "asset", "type", "fc", "official"]],
                      c.assign(asset=c["team"], type="constructor")[["t", "asset", "type", "fc", "official"]]])
    rows["gameday"] = rows["t"].map(D["gameday"])
    rows["residual"] = rows["official"] - rows["fc"]
    rows = rows.dropna(subset=["residual"]).sort_values(["gameday", "type", "asset"])
    rows[["gameday", "asset", "type", "fc", "official", "residual"]].round(2).to_csv("results/forecast_residuals_2026.csv", index=False)
    s = rows.groupby("type")["residual"].agg(["mean", "std", "count"])
    print(s.round(2).to_string())
    print("gamedays:", rows["gameday"].nunique(), "| assets per gameday:", rows.groupby("gameday").size().min(), "-", rows.groupby("gameday").size().max())
    # same-weekend correlation between a constructor's error and its drivers' errors
    w = rows.pivot_table(index="gameday", columns="asset", values="residual")
    cors = [w[t].corr(w[[x for x, tm in GRID_2026.items() if tm == t and x in w]].sum(axis=1)) for t in FP.TEAMS if t in w]
    print("corr(constructor error, its drivers' error sum): median %.2f" % np.nanmedian(cors))
