#!/usr/bin/env python3
"""
Your race weekend in one command (my_weekend.py)
================================================
Reads your team from my_team.json, pulls the latest data and runs the planner.

    python my_weekend.py               # after FP1 / FP2 / FP3: fetch this weekend's practice, plan the next race
    python my_weekend.py --no-fetch    # plan again without downloading
    python my_weekend.py --season      # the rest of the season from your team (chips you still have)
    python my_weekend.py --after-race  # after the Grand Prix: official points, new prices and results
    python my_weekend.py -- --avoid Hamilton --sharp     # anything after -- goes to the planner

The team locks at the start of Qualifying (Sprint weekends: the Sprint), so the last useful run is after FP3.
After you make transfers in the app, update my_team.json (team, bank = cost cap, free transfers, chips left).
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(HERE, "my_team.json")
PLANNER_CHIPS = {"wildcard", "3x_booster", "no_negative", "limitless", "autopilot"}


def load_config(path=CONFIG):
    """Read and check my_team.json; returns (cfg, warnings). Raises ValueError on a team the planner can't use."""
    import f1_planner as FP
    with open(path) as f:
        cfg = json.load(f)
    team = [FP.canon_entity(n) for n in cfg["team"]]
    unknown = [n for n, c in zip(cfg["team"], team) if c not in FP.ENTITIES]
    if unknown:
        raise ValueError(f"not a 2026 driver or constructor: {unknown}")
    if len(set(team)) != 7 or sum(c in FP.DRIVERS for c in team) != 5:
        raise ValueError(f"need 5 different drivers and 2 different constructors, got {team}")
    warnings = []
    chips = list(cfg.get("chips_left", []))
    skipped = [c for c in chips if c not in PLANNER_CHIPS]
    if skipped:
        warnings.append(f"chips the planner does not model, ignored: {skipped} (final_fix: use final_fix.py after Qualifying)")
    cfg["chips_left"] = [c for c in chips if c in PLANNER_CHIPS]
    cfg["team"] = team
    return cfg, warnings


def price_warnings(cfg):
    """Compare the prices you saw in the app with the game's feed (fantasy_f1_prices_official.csv)."""
    import f1_planner as FP
    seen = cfg.get("prices_seen") or {}
    if not seen or not os.path.exists(os.path.join(HERE, "fantasy_f1_prices_official.csv")):
        return []
    now = FP.current_prices(path=os.path.join(HERE, "fantasy_f1_prices_official.csv"))
    out = []
    for name, p in seen.items():
        feed = float(now[FP.ENTITIES.index(FP.canon_entity(name))])
        if abs(feed - float(p)) > 0.05:
            out.append(f"{name}: app ${p}m vs feed ${feed}m")
    return out


def planner_command(cfg, season=False, extra=()):
    script = "plan_season.py" if season else "plan_next_race.py"
    cmd = [sys.executable, script, "--team", *cfg["team"], "--bank", str(cfg["bank"]),
           "--free", str(cfg["free_transfers"])]
    if cfg["chips_left"]:
        cmd += ["--chips", *cfg["chips_left"]]
    return cmd + list(extra)


def run(cmd, dry_run=False):
    print("$ " + " ".join(f'"{c}"' if " " in c else c for c in cmd[1:]), flush=True)
    if not dry_run:
        subprocess.run(cmd, cwd=HERE, check=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-fetch", action="store_true", help="skip downloading, just plan")
    ap.add_argument("--season", action="store_true", help="plan the rest of the season instead of the next race")
    ap.add_argument("--after-race", action="store_true", help="fetch official points / prices and results, no plan")
    ap.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    ap.add_argument("--config", default=CONFIG)
    argv = list(sys.argv[1:] if argv is None else argv)
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    a = ap.parse_args(argv[:argv.index("--")] if "--" in argv else argv)

    cfg, warnings = load_config(a.config)
    print(f"Team ({cfg.get('as_of', 'my_team.json')}): {', '.join(cfg['team'])} | bank ${cfg['bank']}m | "
          f"{cfg['free_transfers']} free transfer(s) | chips left: {cfg['chips_left'] or 'none'}")
    for w in warnings:
        print(f"  note: {w}")

    py = sys.executable
    if a.after_race:
        run([py, "fetch_fantasy_feed.py"], a.dry_run)
        run([py, "fetch_openf1_full.py", "--years", "2026"], a.dry_run)
        print("\nNow update my_team.json with what the app shows (team, cost cap, free transfers, chips left, "
              "prices_seen) and run `python my_weekend.py --no-fetch` for next week.")
        return
    if not a.no_fetch:
        run([py, "fetch_openf1_full.py", "--years", "2026"], a.dry_run)     # this weekend's practice so far
    for w in price_warnings(cfg):
        print(f"  price check: {w} - if the app is newer, run `python fetch_fantasy_feed.py` "
              f"(or pass -- --price NAME=VALUE)")
    run(planner_command(cfg, a.season, extra), a.dry_run)


if __name__ == "__main__":
    main()
