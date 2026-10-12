#!/usr/bin/env python3
"""Regenerate raw_historical_data.json (<=2025) and raw_2026_results.json from the one-pull cache openf1_full/
(no network).  Keeps the pipeline's input in one verified place; see docs/OPENF1_NOTES.md."""
import json

import openf1_cache as C

full = C.load_all()
legacy = C.to_legacy_meetings(full)
hist = [m for m in legacy if m["year"] <= 2025]
cur = [m for m in legacy if m["year"] >= 2026]
json.dump(hist, open("raw_historical_data.json", "w"))
json.dump(cur, open("raw_2026_results.json", "w"))
print(f"raw_historical_data.json: {len(hist)} weekends | raw_2026_results.json: {len(cur)} weekends")
