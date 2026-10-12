"""Decision-level test: play the 2026 season with a forecaster, walk-forward, on the game's official points and prices.
At gameday g the player knows: results and official points of gamedays < g, the real prices for gameday g, and that
weekend's practice (before the lock).  It forecasts the next H raced weekends, solves exactly, plays one week, rolls on.
Compared with the proven hindsight optimum (5741)."""
import os, sys, json, datetime, pickle
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd

H = int(os.environ.get("HORIZON", "3"))
PANEL = os.environ.get("PANEL_PKL")
CACHE = os.environ.get("S26_CACHE")


def _context():
    import mlflow_experiments as X, openf1_cache as C
    from experiments import zoo, s26
    res, truth, ctruth = X.get_data()
    full = C.load_all(only_complete=False)
    return res, truth, full, C.to_practice(full), zoo.track_tables(), s26.prepare(PANEL, CACHE)


_CTX = {}
def ctx():
    if "c" not in _CTX:
        _CTX["c"] = _context()
    return _CTX["c"]


def forecast_at(args):
    """Forecasts (H x 33) made before gameday g for the next H raced weekends, by every named forecaster."""
    g, forecasters = args
    import f1_planner as FP
    from f1_season import truncate
    from f1_weekend_data import Results
    res, truth, full, pdata, tracks, D = ctx()
    w26 = res.weekends[res.weekends.year == 2026].sort_values("t")
    t_g = int(w26.iloc[g - 1]["t"])
    rounds = list(w26["round"].iloc[g - 1:g - 1 + H])
    rt = truncate(res, t_g)
    tt = truth[truth["t"] < t_g]
    op = pd.read_csv("fantasy_f1_official_points.csv")
    op = op[op["gameday"] < g]
    panel_hist = D["panel"][D["panel"]["t"] < t_g]
    sim = FP.sim_forecasts(rt, rounds)
    out = {}
    for name, spec in forecasters.items():
        if spec == "sim":
            out[name] = np.array([sim[r] for r in rounds])
            continue
        if isinstance(spec, dict) and spec.get("kind") == "sim+ewma":
            ew = FP.ewma_official(op, 0.2) if len(op) else pd.Series(np.nan, index=FP.ENTITIES)
            s = np.array([sim[r] for r in rounds])
            e = ew.reindex(FP.ENTITIES).values
            out[name] = np.where(np.isnan(e)[None, :], s, (1 - spec["w"]) * s + spec["w"] * e[None, :])
            continue
        cfg = spec if isinstance(spec, dict) and "specs" in spec else {"specs": spec if isinstance(spec, list) else [spec]}
        out[name] = _forecast_specs(rt, tt, cfg["specs"], rounds, panel_hist, full, pdata, tracks, sim, op if len(op) else None,
                                    cfg.get("c_sim", 0.5), cfg.get("bonus_alpha", 0.3))
    return g, rounds, out


def _forecast_specs(rt, tt, specs, rounds, panel_hist, full, pdata, tracks, sim, op, c_sim=0.5, bonus_alpha=0.3):
    import f1_planner as FP
    if op is None:
        op = pd.DataFrame({"round": [], "gameday": [], "name": [], "type": [], "points": []})
    return FP.forecast_remaining(rt, tt, specs, rounds, panel_hist, full=full, pdata=pdata, tracks=tracks, sim=sim, op=op,
                                 c_sim=c_sim, bonus_alpha=bonus_alpha)


def play(fc_by_g, discount=0.6, rules=None):
    """Sequential play on official points and prices with given forecasts {g: (rounds, H x 33)}."""
    import f1_planner as FP
    from f1_strategy import Rules, State, solve_strategy, evaluate_plan
    from f1_prices import PriceRule
    rules = rules or Rules()
    op = pd.read_csv("fantasy_f1_official_points.csv")
    pr = pd.read_csv("fantasy_f1_prices_official.csv")
    G = int(op["gameday"].max())
    act = {g: op[op.gameday == g].set_index("name")["points"].reindex(FP.ENTITIES).fillna(0).values for g in range(1, G + 1)}
    price = {g: pr[pr.gameday == g].set_index("name")["price"].reindex(FP.ENTITIES).values for g in range(1, G + 2)}
    state = State(prev_team=None, wealth=rules.budget, free_transfers=rules.free_per_round, chips_left=rules.chips)
    total, log = 0.0, []
    for g in range(1, G + 1):
        rounds, pts = fc_by_g[g]
        Hh = min(len(pts), G - g + 1)
        pts = pts[:Hh]
        path = np.array([price[g]] * Hh)                       # future prices unknown: hold flat (play-policy default)
        plan = solve_strategy(pts, path, FP.IS_DRIVER, rules, state, discount ** np.arange(Hh), time_limit=60, mip_rel_gap=1e-6)
        team, boost, chip = plan.teams[0], plan.boosts[0], plan.chips[0]
        r, per, pen, trs, ok = evaluate_plan([team], [boost], [chip], act[g][None, :], price[g][None, :], FP.IS_DRIVER, rules, state)
        assert ok, g
        total += r
        log.append({"gameday": g, "points": per[0], "chip": chip, "transfers": trs[0], "penalty": pen[0],
                    "team": ", ".join(FP.ENTITIES[e] for e in sorted(team)), "boost": FP.ENTITIES[boost]})
        W = state.wealth
        used = 0 if (state.prev_team is None or chip == "wildcard") else min(trs[0], state.free_transfers)
        state = State(prev_team=team, wealth=W + sum(price[g + 1][e] - price[g][e] for e in team),
                      free_transfers=min(rules.max_bank, state.free_transfers - used + rules.free_per_round),
                      chips_left=tuple(c for c in state.chips_left if c != chip))
    return total, pd.DataFrame(log)


if __name__ == "__main__":
    import mlflow_experiments as X
    forecasters = {"sim": "sim", "sim+ewma0.4": {"kind": "sim+ewma", "w": 0.4}}
    for path in sys.argv[1:]:                                   # extra spec files: results/*.json with {"name", "specs", "c_sim", "bonus_alpha"}
        d = json.load(open(path))
        forecasters[os.path.basename(path).replace(".json", "")] = {"specs": d["specs"], "c_sim": d.get("c_sim", 0.5),
                                                                      "bonus_alpha": d.get("bonus_alpha", 0.3)}
    G = 17
    with ProcessPoolExecutor(max_workers=int(os.environ.get("WORKERS", "6"))) as ex:
        res_ = list(ex.map(forecast_at, [(g, forecasters) for g in range(1, G + 1)]))
    pd.to_pickle(res_, f"results/cache/season_replay_forecasts_{datetime.date.today()}.pkl")
    OPT = 5741.0
    rows = []
    mlflow = X._mlflow()
    with mlflow.start_run(run_name=f"s26-season-replay {datetime.date.today()}"):
        mlflow.set_tags({"phase": "s26-season", "data_version": X.DATA_VERSION, "git_sha": X.git_sha(), "horizon": H,
                         "truth": "official points and prices", "optimum": OPT})
        for name in forecasters:
            fc = {g: (rounds, out[name]) for g, rounds, out in res_}
            tot, log = play(fc)
            half = log["points"].values
            rows.append({"forecaster": name, "season_points": tot, "share_of_optimum": tot / OPT,
                         "first_half": float(half[:8].sum()), "second_half": float(half[8:].sum())})
            with mlflow.start_run(run_name=f"season play: {name}", nested=True):
                mlflow.set_tags({"phase": "s26-season", "forecaster": name})
                mlflow.log_metrics({k: v for k, v in rows[-1].items() if k != "forecaster"})
            log.to_csv(f"results/season_play_{name.replace('/', '_').replace('+', '_plus_')}.csv", index=False)
        df = pd.DataFrame(rows).sort_values("season_points", ascending=False)
        df.to_csv("results/s26_season_replay.csv", index=False)
        mlflow.log_artifact("results/s26_season_replay.csv")
    print(df.round(3).to_string(index=False))
