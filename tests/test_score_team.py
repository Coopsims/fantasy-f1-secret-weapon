import numpy as np
import pandas as pd
import pytest

import f1_planner as FP
import score_team as ST

TEAM = ["Hamilton", "Antonelli", "Colapinto", "Lawson", "Bearman", "Ferrari", "Racing Bulls"]


def _points(tmp_path):
    rows = [{"round": 19, "gameday": 17, "name": n, "type": "driver" if n in FP.DRIVERS else "constructor",
             "points": 1.0} for n in FP.ENTITIES]
    pts = {"Hamilton": 22, "Antonelli": 45, "Colapinto": -8, "Lawson": 12, "Bearman": -20, "Ferrari": 50,
           "Racing Bulls": 18}
    rows += [{"round": 20, "gameday": 18, "name": n, "type": "driver" if n in FP.DRIVERS else "constructor",
              "points": float(pts.get(n, 3))} for n in FP.ENTITIES if n != "Colapinto"]   # an absent asset scores 0
    p = tmp_path / "pts.csv"
    p.write_text(pd.DataFrame(rows).to_csv(index=False))
    return str(p)


def test_scores_like_the_game(tmp_path):
    g, race, row = ST.official_row(path=_points(tmp_path))
    assert (g, race) == (18, "US GP") and row[FP.ENTITIES.index("Colapinto")] == 0
    total, lines = ST.score(TEAM, "Antonelli", row)
    assert total == 22 + 2 * 45 + 0 + 12 - 20 + 50 + 18
    assert dict((n, m) for n, m, _ in lines)["Antonelli"] == 2
    assert ST.score(TEAM, "Antonelli", row, chip="no_negative")[0] == total + 20
    assert ST.score(TEAM, "Lawson", row, chip="3x_booster", boost3="Antonelli")[0] == 22 + 3 * 45 + 2 * 12 - 20 + 50 + 18
    assert ST.score(TEAM, "Bearman", row, chip="autopilot")[0] == total            # 2x moves to the top scorer
    assert ST.score(TEAM, "Antonelli", row, penalty=10)[0] == total - 10
    assert ST.official_row(17, _points(tmp_path))[2].sum() == len(FP.ENTITIES)


def test_rejects_bad_teams(tmp_path):
    row = ST.official_row(path=_points(tmp_path))[2]
    with pytest.raises(ValueError):
        ST.score(TEAM[:6] + ["Hulkenberg"], "Antonelli", row)          # 6 drivers
    with pytest.raises(ValueError):
        ST.score(TEAM, "Verstappen", row)                               # boost not in the team
    with pytest.raises(ValueError):
        ST.score(TEAM, "Antonelli", row, chip="3x_booster", boost3="Antonelli")


def test_cli_defaults_to_my_team(tmp_path, capsys):
    total = ST.main(["--points", _points(tmp_path)])
    out = capsys.readouterr().out
    assert "US GP (gameday 18)" in out and "my_team.json" in out
    assert total == pytest.approx(22 + 2 * 45 + 12 - 20 + 50 + 18)
