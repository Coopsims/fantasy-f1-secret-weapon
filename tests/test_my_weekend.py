import json

import pytest

import my_weekend as W
from plan_season import practice_fingerprint


def _cfg(tmp_path, **over):
    cfg = json.load(open(W.CONFIG))
    cfg.update(over)
    p = tmp_path / "team.json"
    p.write_text(json.dumps(cfg))
    return str(p)


def test_my_team_config_is_valid():
    cfg, warnings = W.load_config()
    assert len(cfg["team"]) == 7 and "Racing Bulls" in cfg["team"]
    assert cfg["chips_left"] == ["wildcard"] and not warnings


def test_planner_command_for_next_race_and_season():
    cfg, _ = W.load_config()
    nxt = W.planner_command(cfg, extra=["--sharp"])
    assert nxt[1] == "plan_next_race.py" and nxt[-1] == "--sharp"
    assert nxt[nxt.index("--team") + 1:nxt.index("--bank")] == cfg["team"]
    assert nxt[nxt.index("--bank") + 1] == "2.4" and nxt[nxt.index("--free") + 1] == "3"
    assert nxt[nxt.index("--chips") + 1:nxt.index("--sharp")] == ["wildcard"]
    assert W.planner_command(cfg, season=True)[1] == "plan_season.py"


def test_bad_team_and_unmodelled_chips(tmp_path):
    with pytest.raises(ValueError):
        W.load_config(_cfg(tmp_path, team=["Hamilton", "Antonelli", "Colapinto", "Lawson", "Nobody", "Ferrari", "Audi"]))
    with pytest.raises(ValueError):          # 6 drivers + 1 constructor
        W.load_config(_cfg(tmp_path, team=["Hamilton", "Antonelli", "Colapinto", "Lawson", "Bearman", "Norris", "Audi"]))
    cfg, warnings = W.load_config(_cfg(tmp_path, chips_left=["wildcard", "final_fix"]))
    assert cfg["chips_left"] == ["wildcard"] and "final_fix" in warnings[0]


def test_dry_run_prints_fetch_then_plan(capsys):
    W.main(["--dry-run"])
    out = capsys.readouterr().out
    assert (out.index("fetch_openf1_full.py --years 2026") < out.index("rebuild_raw_from_full.py")
            < out.index("plan_next_race.py --team"))
    W.main(["--dry-run", "--after-race"])
    out = capsys.readouterr().out
    assert out.index("fetch_openf1_full.py") < out.index("rebuild_raw_from_full.py") and "plan_next_race" not in out
    W.main(["--dry-run", "--no-fetch", "--season", "--", "--avoid", "Hamilton"])
    out = capsys.readouterr().out
    assert "fetch_openf1_full" not in out and "plan_season.py" in out and out.strip().endswith("--avoid Hamilton")


def test_practice_fingerprint_changes_when_practice_arrives():
    """The forecast cache must not be reused after FP1: its key includes the practice laps known."""
    before = [{"meeting_key": 1290, "sessions": []}]
    fp1 = [{"meeting_key": 1290, "sessions": [{"session_name": "Practice 1", "laps": [{}] * 400}]}]
    fp2 = [{"meeting_key": 1290, "sessions": [{"session_name": "Practice 1", "laps": [{}] * 400},
                                              {"session_name": "Practice 2", "laps": [{}] * 350}]}]
    keys = [json.dumps(practice_fingerprint(p)) for p in (before, fp1, fp2)]
    assert len(set(keys)) == 3
