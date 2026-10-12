"""How good is the price reconstruction? Checked against the game's official prices for every 2026 gameday
(fantasy_f1_prices_official.csv / fantasy_f1_official_points.csv from fetch_fantasy_feed.py)."""
import os, sys, datetime
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dataclasses import replace
import numpy as np, pandas as pd
import f1_prices as FP
import f1_season as S
from f1_weekend_data import load_results

pd.set_option("display.width", 220)
E = S.ENTITIES
pr = pd.read_csv("fantasy_f1_prices_official.csv")
op = pd.read_csv("fantasy_f1_official_points.csv")
G = int(op["gameday"].max())                                    # completed gamedays (17)
P_off = np.array([[float(pr[(pr.gameday == g) & (pr.name == e)]["price"].iloc[0]) for e in E] for g in range(1, G + 2)])  # (G+1, E): before gameday 1..G+1
pts_off = np.array([[float(op[(op.gameday == g) & (op.name == e)]["points"].iloc[0]) for e in E] for g in range(1, G + 1)])
res = load_results(verbose=False)
act = S.season_actuals(res, 2026)
pts_ours = act["pts"]
assert len(pts_ours) == G, (len(pts_ours), G)
real_moves = np.round(np.diff(P_off, axis=0), 1)
print(f"{G} transitions x {len(E)} assets; real move sizes:", pd.Series(real_moves.ravel()).value_counts().sort_index().to_dict())
print("lowest real prices:", sorted(set(np.round(P_off.min(0), 1)))[:4], "(the code assumed a $4.0m floor)")

def path_errors(P, label):
    err = np.abs(P[:G + 1] - P_off)
    by = pd.DataFrame({"after_gameday": range(G + 1), "mae": err.mean(1), "exact": (err < 0.05).mean(1), "max": err.max(1)})
    print(f"\n{label}: MAE by gameday", by.mae.round(2).tolist())
    print(f"   exact share {by.exact.mean():.0%} overall; after Singapore (gameday {G}): MAE ${by.mae.iloc[-1]:.2f}m, max ${by['max'].iloc[-1]:.1f}m")
    return by

# A. what the pipeline used until now: rule fitted to your round 0-2 prices, driven by OUR reconstructed points
mine = pd.read_csv("fantasy_f1_prices.csv"); mine["name"] = mine["name"].map(S._canon_entity)
anch = FP.checkpoints_from_csv(mine, act["rounds"], E)
base = np.array([S.DEFAULT_DRIVER_PRICES[d] for d in S.DRIVERS] + [S.DEFAULT_CONSTRUCTOR_PRICES[t] for t in S.TEAMS])
PA, infoA = FP.reconstruct(pts_ours, base, anch, verbose=False)
path_errors(PA, "A  pipeline (fitted to rounds 0-2, our points)")
# B. same rule, official points, official start prices: isolates the rule from our points
PB = FP.simulate(pts_off, P_off[0], infoA["rule"])
path_errors(PB, "B  same fitted rule, OFFICIAL points")
PB2 = FP.simulate(pts_off, P_off[0], replace(FP.PriceRule(), floor=3.0))
path_errors(PB2, "B2 guide's default rule (floor 3.0), OFFICIAL points")

# C. one-step accuracy from the real prices: is the rule itself right?
def one_step(rule, pts):
    rows = []
    for j in range(G):
        win = [pts[i] / np.maximum(P_off[i], 0.1) for i in range(max(0, j - rule.window + 1), j + 1)]
        pred = FP.step_prices(P_off[j], win, rule)
        rows.append(np.mean(np.abs(pred - P_off[j + 1]) < 0.05))
    return np.array(rows)
for lbl, rule, pts in (("guide default, floor 3", replace(FP.PriceRule(), floor=3.0), pts_off),
                       ("fitted on rounds 0-2", replace(infoA["rule"], floor=3.0), pts_off),
                       ("guide default, OUR points", replace(FP.PriceRule(), floor=3.0), pts_ours)):
    r = one_step(rule, pts)
    print(f"\nC  one-step exact share, {lbl}: mean {r.mean():.0%}; by gameday {np.round(r, 2).tolist()}")

# D. fit the rule to official prices: first half -> second half (honest), then all
anchors_all = {j: P_off[j] for j in range(G + 1)}
half = G // 2
r1, l1 = FP.fit_rule(pts_off, {j: P_off[j] for j in range(half + 1)}, base=replace(FP.PriceRule(), floor=3.0))
print(f"\nD  fitted on gamedays 1-{half}: {r1}")
r_h = one_step(r1, pts_off)
print(f"   one-step exact share on gamedays {half + 1}-{G} (not used for fitting): {r_h[half:].mean():.0%}  (guide default there: {one_step(replace(FP.PriceRule(), floor=3.0), pts_off)[half:].mean():.0%})")
rall, lall = FP.fit_rule(pts_off, anchors_all, base=replace(FP.PriceRule(), floor=3.0))
print(f"   fitted on all {G}: {rall}; one-step exact {one_step(rall, pts_off).mean():.0%}")

# E. where does the default rule miss? (tier, rating)
rule = replace(FP.PriceRule(), floor=3.0)
miss = []
for j in range(G):
    win = [pts_off[i] / np.maximum(P_off[i], 0.1) for i in range(max(0, j - 2), j + 1)]
    ppm = np.mean(win, 0)
    pred = FP.step_prices(P_off[j], win, rule)
    for e in range(len(E)):
        if abs(pred[e] - P_off[j + 1][e]) >= 0.05:
            miss.append((j + 1, E[e], P_off[j][e], round(ppm[e], 3), round(pred[e] - P_off[j][e], 1), round(P_off[j + 1][e] - P_off[j][e], 1)))
m = pd.DataFrame(miss, columns=["after_gameday", "asset", "price", "ppm_3race", "rule_move", "real_move"])
print(f"\nE  default-rule misses: {len(m)} of {G * len(E)}")
print(m.sort_values("ppm_3race").to_string(index=False, max_rows=40))
