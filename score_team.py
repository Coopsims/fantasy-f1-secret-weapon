#!/usr/bin/env python3
"""
What a team scored (score_team.py)
==================================
Any team's official points for a completed gameday, from the game's feed (fantasy_f1_official_points.csv: run
`python my_weekend.py --after-race` or `python fetch_fantasy_feed.py` first).  Default: the team and boost in
my_team.json at the latest gameday - before you update the file, that is the team you had before your transfers.

    python score_team.py                                  # my_team.json's team and boost, latest gameday
    python score_team.py --team Antonelli Hulkenberg Lindblad Lawson Bearman Ferrari Mercedes --boost Antonelli
    python score_team.py --gameday 17 --chip no_negative --penalty 10
    python score_team.py --chip 3x_booster --boost3 Antonelli --boost Lawson
"""
import argparse

import numpy as np
import pandas as pd

import f1_planner as FP
from f1_calendar import RACE_CALENDAR_2026

POINTS_CSV = "fantasy_f1_official_points.csv"


def official_row(gameday=None, path=POINTS_CSV):
    """(gameday, race name, official points of every asset in FP.ENTITIES order; 0 if it did not take part)."""
    op = pd.read_csv(path)
    g = int(op["gameday"].max()) if gameday is None else int(gameday)
    wk = op[op["gameday"] == g]
    if wk.empty:
        raise SystemExit(f"no official points for gameday {g} in {path} (latest: {int(op['gameday'].max())})")
    row = np.zeros(len(FP.ENTITIES))
    for r in wk.itertuples():
        c = FP.canon_entity(r.name)
        if c in FP.ENTITIES:
            row[FP.ENTITIES.index(c)] = float(r.points)
    rnd = int(wk["round"].iloc[0])
    name = next((r["name"] for r in RACE_CALENDAR_2026 if r["round"] == rnd), f"round {rnd}")
    return g, name, row


def score(team, boost, pts_row, chip=None, boost3=None, penalty=0.0):
    """The week scored as the game does: (total, [(asset, multiplier, points it adds)]).
    chip: None, 'no_negative', '3x_booster' (boost3 = the x3 driver, the x2 stays on boost) or 'autopilot' (the x2
    goes to the team's top-scoring driver)."""
    idx = [FP.ENTITIES.index(FP.canon_entity(n)) for n in team]
    unknown = [n for n in team if FP.canon_entity(n) not in FP.ENTITIES]
    if unknown or len(set(idx)) != 7 or sum(FP.IS_DRIVER[e] for e in idx) != 5:
        raise ValueError(f"need 5 different drivers and 2 different constructors, got {list(team)}")
    if chip == "autopilot":
        b = max((e for e in idx if FP.IS_DRIVER[e]), key=lambda e: pts_row[e])
        chip = None
    else:
        b = FP.ENTITIES.index(FP.canon_entity(boost))
    b3 = FP.ENTITIES.index(FP.canon_entity(boost3)) if chip == "3x_booster" else None
    if b not in idx or not FP.IS_DRIVER[b] or (chip == "3x_booster" and (b3 is None or b3 == b or b3 not in idx
                                                                        or not FP.IS_DRIVER[b3])):
        raise ValueError("the 2x (and the x3) must be different drivers in the team")
    x = np.maximum(pts_row, 0.0) if chip == "no_negative" else np.asarray(pts_row, float)
    lines = [(FP.ENTITIES[e], 3 if e == b3 else 2 if e == b else 1, float(x[e]) * (3 if e == b3 else 2 if e == b else 1))
             for e in sorted(idx, key=lambda e: (not FP.IS_DRIVER[e], -x[e]))]
    total = float(FP.week_points_draws(frozenset(idx), b, chip, pts_row, np.zeros((1, len(pts_row))), penalty, b3)[0])
    assert abs(total - (sum(p for _, _, p in lines) - penalty)) < 1e-9
    return total, lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--team", nargs=7, metavar="NAME", help="5 drivers + 2 constructors (default: my_team.json)")
    ap.add_argument("--boost", help="the 2x driver (default: my_team.json's boost)")
    ap.add_argument("--chip", choices=["no_negative", "3x_booster", "autopilot"])
    ap.add_argument("--boost3", help="with --chip 3x_booster: the x3 driver")
    ap.add_argument("--penalty", type=float, default=0.0, help="transfer penalty that week (points)")
    ap.add_argument("--gameday", type=int, help="default: the latest in the feed")
    ap.add_argument("--points", default=POINTS_CSV)
    a = ap.parse_args(argv)

    if a.team is None or a.boost is None:
        import my_weekend
        cfg, _ = my_weekend.load_config()
        team, boost = a.team or cfg["team"], a.boost or cfg.get("boost")
        src = f"my_team.json ({cfg.get('as_of', '')})"
    else:
        team, boost, src = a.team, a.boost, "--team"
    g, race, row = official_row(a.gameday, a.points)
    total, lines = score(team, boost, row, a.chip, a.boost3, a.penalty)
    print(f"{race} (gameday {g}), official points - team from {src}:")
    for name, mult, pts in lines:
        print(f"  {name:16s} {'' if mult == 1 else f'{mult}x':3s} {pts:6.0f}")
    if a.penalty:
        print(f"  {'transfer penalty':20s} {-a.penalty:6.0f}")
    print(f"  {'total':20s} {total:6.0f}" + (f"   ({a.chip})" if a.chip else ""))
    return total


if __name__ == "__main__":
    main()
