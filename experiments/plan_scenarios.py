"""Season plans for the user's team under each chip scenario (the chips left are unknown) -> results/SEASON_PLAN.md."""
import os, sys, datetime
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, pandas as pd
import f1_planner as FP
from plan_season import forecasts, load_specs

TEAM = ["Hamilton", "Antonelli", "Colapinto", "Lawson", "Bearman", "Ferrari", "Racing Bulls"]
BANK, FREE = 2.4, 3
SCENARIOS = [(), ("3x_booster",), ("no_negative",), ("wildcard",), ("limitless",),
             ("wildcard", "3x_booster", "no_negative", "limitless")]
NOTE = ("No Negative is valued with the forecast error (E[max(X,0)] - E[X], sd 14.6 per driver / 20.6 per constructor); "
        "Autopilot (not optimised): about 6 points in a typical week - play it in a week with no other chip. Final Fix "
        "is not modelled (it needs qualifying results).")

if __name__ == "__main__":
    import mlflow_experiments as X
    res, _, _ = X.get_data()
    rounds = FP.upcoming_rounds(res)
    names = [next(r["name"] for r in FP.RACE_CALENDAR_2026 if r["round"] == x) for x in rounds]
    specs, cfg, spec_name = load_specs(FP.SPEC_FILE)
    pts = forecasts(rounds, specs, cfg)
    price = FP.current_prices()
    path = FP.project_prices(price, pts, FP.official_tables())
    lines = [f"# Rest-of-season plans for your team ({datetime.datetime.now():%Y-%m-%d %H:%M})", "",
             f"Model: {spec_name} (planning blend on). Start: {', '.join(TEAM)}, bank ${BANK}m, {FREE} free transfers. "
             "Prices: the game's current prices, projected with its price rule. Expected points; re-plan every week.", "", NOTE, ""]
    summary = []
    for chips in SCENARIOS:
        p, st = FP.plan(pts, path, team=TEAM, bank=BANK, free=FREE, chips=chips, discount=0.9)
        summary.append((", ".join(chips) or "no chips", p.total, p.chips))
        lines += [f"## Chips available: {', '.join(chips) or 'none'} - expected {p.total:.0f} points over {len(rounds)} races", "",
                  "| race | drivers | constructors | boost | chip | transfers in | out | exp. pts |", "|---|---|---|---|---|---|---|---|"]
        prev = set(FP.canon_entity(n) for n in TEAM)
        for k, nm in enumerate(names):
            team = [FP.ENTITIES[e] for e in sorted(p.teams[k])]
            din, dout = sorted(set(team) - prev), sorted(prev - set(team))
            lines.append(f"| {nm} | {', '.join(x for x in team if x in FP.DRIVERS)} | {', '.join(x for x in team if x in FP.TEAMS)} | "
                         f"{FP.boost_label(p, k)} | {p.chips[k] or ''} | {', '.join(din) or '-'} | {', '.join(dout) or '-'} | {p.round_points[k]:.1f} |")
            if p.chips[k] != "limitless":
                prev = set(team)
        lines.append("")
    lines[6:6] = ["| chips available | expected points, rest of season | chips used (race by race) |", "|---|---|---|"] + \
                 [f"| {c} | {t:.0f} | {', '.join(f'{names[i]}: {x}' for i, x in enumerate(ch) if x) or '-'} |" for c, t, ch in summary] + [""]
    open("results/SEASON_PLAN.md", "w").write("\n".join(lines))
    print("\n".join(lines[:16]))
