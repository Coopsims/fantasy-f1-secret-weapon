#!/usr/bin/env python3
"""
F1 Fantasy Team Optimizer (fantasy_optimizer.py)
=================================================
Interactive tool for managing your Fantasy F1 2026 team.

Features:
- Enter your current team, budget, transfers, and available chips
- Uses ML predictions to recommend optimal transfers
- Decides whether to save transfers, use both, or activate a chip
- Tracks actual results per weekend in a CSV history file

Usage:
    source venv/bin/activate
    python fantasy_optimizer.py

Chips available in Fantasy F1 2026:
- Wildcard: Unlimited free transfers for one race (use wisely, limited per season)
- 3x Booster (Mega Driver): Triple one driver's score instead of 2x
- No Negative: All negative driver scores become 0 for the weekend
- Final Fix: Extra free transfer after qualifying (limited use)
- Autopilot: Auto-picks optimal team based on free practice data
"""

import csv
import os
import re
import time
from itertools import combinations
from datetime import datetime

import pandas as pd
import requests

from f1_ml_predictor import (
    predict_driver_scores,
    predict_constructor_scores,
    predict_driver_std,
    get_weekend_forecast,
    fetch_known_results,
    register_known_results,
    get_race_info,
    get_actual_round_points,
    get_all_completed_points,
    RACE_CALENDAR_2026,
)
from f1_calendar import GRID_2026

# ──────────────────────────────────────────────────────────────
# Starting 2026 Driver & Constructor Prices (pre-season defaults)
# ──────────────────────────────────────────────────────────────

_DEFAULT_DRIVER_PRICES = {
    "Verstappen": 27.7, "Russell": 27.4, "Norris": 27.2, "Piastri": 25.5,
    "Antonelli": 23.2, "Leclerc": 22.8, "Hamilton": 22.5, "Hadjar": 15.1,
    "Gasly": 12.0, "Sainz": 11.8, "Albon": 11.6, "Alonso": 10.0,
    "Stroll": 8.0, "Bearman": 7.4, "Ocon": 7.3, "Hulkenberg": 6.8,
    "Lawson": 6.5, "Bortoleto": 6.4, "Lindblad": 6.2, "Colapinto": 6.2,
    "Perez": 6.0, "Bottas": 5.9,
}

_DEFAULT_CONSTRUCTOR_PRICES = {
    "Mercedes": 29.3, "McLaren": 28.7, "Red Bull Racing": 28.2,
    "Ferrari": 23.3, "Alpine": 12.5, "Williams": 12.0,
    "Aston Martin": 9.1, "Haas F1 Team": 7.4, "Audi": 5.4,
    "Racing Bulls": 6.3, "Cadillac": 4.8,
}

# Mapping from f1fantasytools.com IDs to our names
_TOOLS_DRIVER_ID_MAP = {
    "RED_VER": "Verstappen", "MER_RUS": "Russell", "MCL_NOR": "Norris",
    "MCL_PIA": "Piastri", "MER_ANT": "Antonelli", "FER_LEC": "Leclerc",
    "FER_HAM": "Hamilton", "RED_HAD": "Hadjar", "ALP_GAS": "Gasly",
    "WIL_SAI": "Sainz", "WIL_ALB": "Albon", "AST_ALO": "Alonso",
    "HAA_BEA": "Bearman", "HAA_OCO": "Ocon", "AST_STR": "Stroll",
    "AUD_BOR": "Bortoleto", "VRB_LIN": "Lindblad", "ALP_COL": "Colapinto",
    "VRB_LAW": "Lawson", "AUD_HUL": "Hulkenberg", "CAD_PER": "Perez",
    "CAD_BOT": "Bottas",
}
_TOOLS_CONSTRUCTOR_ID_MAP = {
    "ALP": "Alpine", "AST": "Aston Martin", "AUD": "Audi", "CAD": "Cadillac",
    "FER": "Ferrari", "HAA": "Haas", "MCL": "McLaren",
    "MER": "Mercedes", "RED": "Red Bull Racing", "VRB": "Racing Bulls",
    "WIL": "Williams",
}

# Price change amounts by tier and level
_PRICE_CHANGE = {
    "A Tier": {"GREAT": 0.3, "GOOD": 0.1, "POOR": -0.1, "TERRIBLE": -0.3},
    "B Tier": {"GREAT": 0.6, "GOOD": 0.2, "POOR": -0.2, "TERRIBLE": -0.6},
}

HISTORY_FILE = "fantasy_f1_history.csv"
PRICE_HISTORY_FILE = "fantasy_f1_prices.csv"
TRADE_PENALTY_PTS = 10  # Points deducted per extra trade beyond free allowance

# Weight for price-change value in the composite optimiser score.
# 1 point of predicted price increase (in $m) is worth this many fantasy points.
# E.g. 10 means +$0.3m price rise is valued like +3 fantasy points.
PRICE_VALUE_WEIGHT = 10.0


# ──────────────────────────────────────────────────────────────
# Price Tracking
# ──────────────────────────────────────────────────────────────

def _init_price_history():
    """Create the price history CSV with pre-season defaults if it doesn't exist."""
    if os.path.exists(PRICE_HISTORY_FILE):
        return
    rows = []
    for name, price in _DEFAULT_DRIVER_PRICES.items():
        rows.append({"round": 0, "date": "2026-01-01", "name": name,
                     "type": "driver", "price": price})
    for name, price in _DEFAULT_CONSTRUCTOR_PRICES.items():
        rows.append({"round": 0, "date": "2026-01-01", "name": name,
                     "type": "constructor", "price": price})
    df = pd.DataFrame(rows)
    df.to_csv(PRICE_HISTORY_FILE, index=False)
    print(f"  ✅ Initialised {PRICE_HISTORY_FILE} with pre-season prices (Round 0).")


def load_current_prices():
    """
    Load the latest driver and constructor prices from the price history CSV.
    Falls back to hardcoded defaults if the file doesn't exist.
    Returns (driver_prices_dict, constructor_prices_dict).
    """
    _init_price_history()
    df = pd.read_csv(PRICE_HISTORY_FILE)
    # Get the most recent entry per name
    latest = df.sort_values("round").drop_duplicates(subset="name", keep="last")
    driver_prices = {}
    constructor_prices = {}
    for _, row in latest.iterrows():
        if row["type"] == "driver":
            driver_prices[row["name"]] = float(row["price"])
        else:
            constructor_prices[row["name"]] = float(row["price"])
    # Fill any missing with defaults
    for name, price in _DEFAULT_DRIVER_PRICES.items():
        if name not in driver_prices:
            driver_prices[name] = price
    for name, price in _DEFAULT_CONSTRUCTOR_PRICES.items():
        if name not in constructor_prices:
            constructor_prices[name] = price
    return driver_prices, constructor_prices


def get_price_round():
    """Return the highest round number currently stored in the price history."""
    if not os.path.exists(PRICE_HISTORY_FILE):
        return 0
    df = pd.read_csv(PRICE_HISTORY_FILE)
    return int(df["round"].max()) if not df.empty else 0


def fetch_f1fantasytools_prices():
    """
    Scrape current driver & constructor prices from f1fantasytools.com.
    Returns (driver_prices, constructor_prices) dicts, or (None, None) on failure.
    """
    try:
        resp = requests.get("https://f1fantasytools.com/budget-builder", timeout=15)
        resp.raise_for_status()
        html = resp.text.replace('\\"', '"').replace('\\n', '\n')

        driver_prices = {}
        for m in re.finditer(
            r'"id":"([^"]+)","abbreviation":"[^"]+","type":"driver",'
            r'"isActive":(true|false),"color":"[^"]*","price":([0-9.]+)',
            html,
        ):
            did, _active, price = m.groups()
            name = _TOOLS_DRIVER_ID_MAP.get(did)
            if name:
                driver_prices[name] = float(price)

        constructor_prices = {}
        for m in re.finditer(
            r'"id":"([^"]+)","abbreviation":"[^"]+","type":"constructor",'
            r'"color":"[^"]*","price":([0-9.]+)',
            html,
        ):
            cid, price = m.groups()
            name = _TOOLS_CONSTRUCTOR_ID_MAP.get(cid)
            if name:
                constructor_prices[name] = float(price)

        if driver_prices and constructor_prices:
            return driver_prices, constructor_prices
        return None, None

    except Exception as e:
        print(f"  ⚠ Could not fetch prices from f1fantasytools.com: {e}")
        return None, None


def fetch_f1fantasytools_thresholds():
    """
    Scrape price-change thresholds and tiers from f1fantasytools.com.
    Returns list of dicts with keys: name, type, tier, thresholds.
    """
    try:
        resp = requests.get("https://f1fantasytools.com/budget-builder", timeout=15)
        resp.raise_for_status()
        html = resp.text.replace('\\"', '"').replace('\\n', '\n')

        results = []

        # Drivers
        for m in re.finditer(
            r'"id":"([^"]+)","abbreviation":"[^"]+","type":"driver",'
            r'"isActive":[^,]+,"color":"[^"]*","price":([0-9.]+),'
            r'"percentOwned":[^,]+,"x2PercentOwned":[^,]+,'
            r'"priorRacePoints":(-?[0-9.]+|"[^"]*"),'
            r'"priceTier":"([^"]+)",'
            r'"minRequiredPoints":\{([^}]+)\}',
            html,
        ):
            did, price, prior_pts, tier, thresholds_str = m.groups()
            name = _TOOLS_DRIVER_ID_MAP.get(did)
            if name:
                thresholds = {}
                for t in re.finditer(r'"(\w+)":(-?[0-9.]+|"[^"]*")', thresholds_str):
                    level, val = t.groups()
                    try:
                        thresholds[level] = float(val.strip('"').replace('$', ''))
                    except ValueError:
                        thresholds[level] = 0.0
                results.append({
                    "name": name, "type": "driver", "tier": tier,
                    "thresholds": thresholds, "price": float(price),
                })

        # Constructors
        for m in re.finditer(
            r'"id":"([^"]+)","abbreviation":"[^"]+","type":"constructor",'
            r'"color":"[^"]*","price":([0-9.]+),'
            r'"percentOwned":[^,]+,'
            r'"priorRacePoints":(-?[0-9.]+|"[^"]*"),'
            r'"priceTier":"([^"]+)",'
            r'"minRequiredPoints":\{([^}]+)\}',
            html,
        ):
            cid, price, prior_pts, tier, thresholds_str = m.groups()
            name = _TOOLS_CONSTRUCTOR_ID_MAP.get(cid)
            if name:
                thresholds = {}
                for t in re.finditer(r'"(\w+)":(-?[0-9.]+|"[^"]*")', thresholds_str):
                    level, val = t.groups()
                    try:
                        thresholds[level] = float(val.strip('"').replace('$', ''))
                    except ValueError:
                        thresholds[level] = 0.0
                results.append({
                    "name": name, "type": "constructor", "tier": tier,
                    "thresholds": thresholds, "price": float(price),
                })

        return results if results else None

    except Exception as e:
        print(f"  ⚠ Could not fetch thresholds: {e}")
        return None


def fetch_price_change_estimates(driver_scores, constructor_scores):
    """
    Fetch thresholds from f1fantasytools.com and estimate the expected price
    change for every driver/constructor based on *predicted* fantasy points.

    Returns (driver_price_changes, constructor_price_changes) dicts mapping
    name → expected $ change (e.g. +0.3, -0.6, 0.0).
    Returns ({}, {}) if thresholds can't be fetched.
    """
    thresholds = fetch_f1fantasytools_thresholds()
    if not thresholds:
        return {}, {}

    driver_changes = {}
    constructor_changes = {}

    for entry in thresholds:
        name = entry["name"]
        tier = entry["tier"]
        thr = entry["thresholds"]  # e.g. {"TERRIBLE": 1, "POOR": 2, "GOOD": 6, "GREAT": 10}

        if entry["type"] == "driver":
            predicted = driver_scores.get(name, 0.0)
        else:
            predicted = constructor_scores.get(name, 0.0)

        # Determine which level the predicted score would land in.
        # Thresholds are minimums: score >= GREAT → GREAT, >= GOOD → GOOD, etc.
        # If score < TERRIBLE threshold → TERRIBLE (price drops most)
        level = "TERRIBLE"
        for lvl in ["TERRIBLE", "POOR", "GOOD", "GREAT"]:
            if lvl in thr and predicted >= thr[lvl]:
                level = lvl

        change = _PRICE_CHANGE.get(tier, _PRICE_CHANGE["B Tier"]).get(level, 0.0)

        if entry["type"] == "driver":
            driver_changes[name] = change
        else:
            constructor_changes[name] = change

    return driver_changes, constructor_changes


def estimate_price_change(fantasy_pts, all_pts, current_price, entity_type="driver"):
    """
    Estimate a price change using the tier-based model that F1 Fantasy actually uses.

    Price tiers:
      A Tier (expensive): GREAT=+$0.3, GOOD=+$0.1, POOR=-$0.1, TERRIBLE=-$0.3
      B Tier (cheap):     GREAT=+$0.6, GOOD=+$0.2, POOR=-$0.2, TERRIBLE=-$0.6

    The level is determined by where the entity's fantasy points fall
    relative to the field using percentile thresholds.

    Parameters:
        fantasy_pts: This entity's fantasy points for the round
        all_pts: List of ALL entities' fantasy points for the round (same type)
        current_price: Current price of the entity
        entity_type: "driver" or "constructor"

    Returns:
        Estimated new price (rounded to 1 decimal)
    """
    import numpy as np

    if not all_pts:
        return current_price

    # Determine tier based on price
    if entity_type == "driver":
        tier = "A Tier" if current_price >= 20.0 else "B Tier"
    else:
        tier = "A Tier" if current_price >= 20.0 else "B Tier"

    # Determine level based on percentile rank in the field
    sorted_pts = sorted(all_pts)
    rank = sum(1 for p in sorted_pts if p <= fantasy_pts) / len(sorted_pts)

    if rank >= 0.75:
        level = "GREAT"
    elif rank >= 0.50:
        level = "GOOD"
    elif rank >= 0.25:
        level = "POOR"
    else:
        level = "TERRIBLE"

    change = _PRICE_CHANGE[tier].get(level, 0.0)
    new_price = round(current_price + change, 1)
    new_price = max(4.0, new_price)  # Floor

    return new_price


def estimate_all_prices(round_num):
    """
    Auto-estimate prices for all drivers and constructors after a completed round.
    Uses actual fantasy points from OpenF1 results.

    Returns (driver_price_estimates, constructor_price_estimates) as dicts,
    or (None, None) if the round's results aren't available.
    """
    driver_pts, constructor_pts = get_actual_round_points(round_num)
    if driver_pts is None:
        return None, None

    driver_prices, constructor_prices = load_current_prices()

    # Estimate driver price changes
    all_driver_pts = list(driver_pts.values())
    new_driver_prices = {}
    for name, current in driver_prices.items():
        pts = driver_pts.get(name, 0.0)
        new_driver_prices[name] = estimate_price_change(
            pts, all_driver_pts, current, "driver"
        )

    # Estimate constructor price changes
    all_constructor_pts = list(constructor_pts.values())
    new_constructor_prices = {}
    for name, current in constructor_prices.items():
        pts = constructor_pts.get(name, 0.0)
        new_constructor_prices[name] = estimate_price_change(
            pts, all_constructor_pts, current, "constructor"
        )

    return new_driver_prices, new_constructor_prices


def auto_update_prices_to_round(target_round):
    """
    Automatically estimate and save prices for all rounds from the last
    recorded round up to (and including) target_round.
    Processes each round sequentially so cumulative changes are tracked.
    """
    _init_price_history()
    last_round = get_price_round()

    if target_round <= last_round:
        print(f"  ℹ Prices already recorded through Round {last_round}.")
        return

    print(f"\n  🔄 Auto-estimating prices from Round {last_round + 1} to Round {target_round}...")

    for rnd in range(last_round + 1, target_round + 1):
        new_d, new_c = estimate_all_prices(rnd)
        if new_d is None:
            print(f"  ⚠ Round {rnd} results not available yet — stopping at Round {rnd - 1}.")
            break

        today = datetime.now().strftime("%Y-%m-%d")
        rows = []
        for name, price in new_d.items():
            rows.append({"round": rnd, "date": today, "name": name,
                         "type": "driver", "price": price})
        for name, price in new_c.items():
            rows.append({"round": rnd, "date": today, "name": name,
                         "type": "constructor", "price": price})

        new_df = pd.DataFrame(rows)
        new_df.to_csv(PRICE_HISTORY_FILE, mode="a", header=False, index=False)
        print(f"  ✅ Round {rnd} prices estimated and saved.")

        # Reload so next iteration uses updated prices
        global DRIVER_PRICES, CONSTRUCTOR_PRICES
        DRIVER_PRICES, CONSTRUCTOR_PRICES = load_current_prices()

    _show_price_changes(min(target_round, get_price_round()))


def update_prices():
    """
    Update prices after a race weekend.
    Offers: scrape from f1fantasytools.com (recommended), auto-estimate, or manual.
    """
    _init_price_history()
    driver_prices, constructor_prices = load_current_prices()
    last_round = get_price_round()

    print("\n" + "=" * 60)
    print("  💲 UPDATE PRICES AFTER RACE WEEKEND")
    print("=" * 60)
    print(f"  (Prices last recorded after Round {last_round})")

    round_num = int(input("\n  Which round did these prices update after? ").strip())

    print("\n  How would you like to update?")
    print("    1. Scrape live prices from f1fantasytools.com (recommended)")
    print("    2. Auto-estimate from actual race results (tier-based model)")
    print("    3. Enter prices manually")
    mode = input("  → ").strip()

    if mode == "1":
        print("\n  🌐 Fetching current prices from f1fantasytools.com...")
        new_d, new_c = fetch_f1fantasytools_prices()
        if new_d is None:
            print("  ⚠ Scraping failed. Falling back to auto-estimate.")
            auto_update_prices_to_round(round_num)
        else:
            today = datetime.now().strftime("%Y-%m-%d")
            rows = []
            for name, price in new_d.items():
                rows.append({"round": round_num, "date": today, "name": name,
                             "type": "driver", "price": price})
            for name, price in new_c.items():
                rows.append({"round": round_num, "date": today, "name": name,
                             "type": "constructor", "price": price})
            new_df = pd.DataFrame(rows)
            new_df.to_csv(PRICE_HISTORY_FILE, mode="a", header=False, index=False)
            print(f"  ✅ Live prices for Round {round_num} saved ({len(new_d)} drivers, {len(new_c)} constructors)")
            _show_price_changes(round_num)
        # Offer manual overrides
        override = input("\n  Would you like to manually correct any prices? (y/n) ").strip().lower()
        if override == "y":
            _manual_price_overrides(round_num)
        return

    if mode == "2":
        # Auto-estimate all rounds from last_round+1 through round_num
        auto_update_prices_to_round(round_num)
        # Offer manual overrides
        override = input("\n  Would you like to manually correct any prices? (y/n) ").strip().lower()
        if override == "y":
            _manual_price_overrides(round_num)
        return

    # Manual entry (original behavior)
    today = datetime.now().strftime("%Y-%m-%d")
    rows = []

    print("\n  DRIVER PRICES  (press Enter to keep current value)")
    for name in sorted(driver_prices.keys(), key=lambda n: -driver_prices[n]):
        current = driver_prices[name]
        raw = input(f"    {name:15s} [${current:.1f}m] → $").strip()
        new_price = float(raw) if raw else current
        rows.append({"round": round_num, "date": today, "name": name,
                     "type": "driver", "price": new_price})

    print("\n  CONSTRUCTOR PRICES  (press Enter to keep current value)")
    for name in sorted(constructor_prices.keys(), key=lambda n: -constructor_prices[n]):
        current = constructor_prices[name]
        raw = input(f"    {name:20s} [${current:.1f}m] → $").strip()
        new_price = float(raw) if raw else current
        rows.append({"round": round_num, "date": today, "name": name,
                     "type": "constructor", "price": new_price})

    new_df = pd.DataFrame(rows)
    new_df.to_csv(PRICE_HISTORY_FILE, mode="a", header=False, index=False)

    print(f"\n  ✅ Prices for Round {round_num} saved to {PRICE_HISTORY_FILE}")
    _show_price_changes(round_num)


def _manual_price_overrides(round_num):
    """Allow user to manually correct specific auto-estimated prices."""
    driver_prices, constructor_prices = load_current_prices()
    today = datetime.now().strftime("%Y-%m-%d")

    print("\n  Enter corrections (press Enter to keep auto-estimate):")
    rows = []
    changed = False

    print("\n  DRIVERS:")
    for name in sorted(driver_prices.keys(), key=lambda n: -driver_prices[n]):
        current = driver_prices[name]
        raw = input(f"    {name:15s} [${current:.1f}m] → $").strip()
        if raw:
            new_price = float(raw)
            rows.append({"round": round_num, "date": today, "name": name,
                         "type": "driver", "price": new_price})
            changed = True

    print("\n  CONSTRUCTORS:")
    for name in sorted(constructor_prices.keys(), key=lambda n: -constructor_prices[n]):
        current = constructor_prices[name]
        raw = input(f"    {name:20s} [${current:.1f}m] → $").strip()
        if raw:
            new_price = float(raw)
            rows.append({"round": round_num, "date": today, "name": name,
                         "type": "constructor", "price": new_price})
            changed = True

    if changed:
        new_df = pd.DataFrame(rows)
        new_df.to_csv(PRICE_HISTORY_FILE, mode="a", header=False, index=False)
        print(f"  ✅ Manual corrections saved.")
    else:
        print("  ℹ No changes made.")


def _show_price_changes(round_num):
    """Show price changes between the two most recent rounds."""
    df = pd.read_csv(PRICE_HISTORY_FILE)
    rounds = sorted(df["round"].unique())
    if len(rounds) < 2:
        return
    prev_round = rounds[-2] if rounds[-1] == round_num else rounds[-1]
    prev = df[df["round"] == prev_round].set_index("name")["price"]
    curr = df[df["round"] == round_num].set_index("name")["price"]
    common = set(prev.index) & set(curr.index)
    if not common:
        return
    print(f"\n  📈 Price changes (Round {prev_round} → Round {round_num}):")
    changes = []
    for name in common:
        diff = curr[name] - prev[name]
        if abs(diff) > 0.01:
            changes.append((name, prev[name], curr[name], diff))
    changes.sort(key=lambda x: -abs(x[3]))
    if not changes:
        print("    No changes.")
        return
    for name, old, new, diff in changes:
        arrow = "↑" if diff > 0 else "↓"
        print(f"    {name:20s}  ${old:.1f}m → ${new:.1f}m  ({arrow} ${abs(diff):.1f}m)")


def show_price_history():
    """Display full price history across all rounds."""
    if not os.path.exists(PRICE_HISTORY_FILE):
        print("  No price history yet.")
        return
    df = pd.read_csv(PRICE_HISTORY_FILE)
    rounds = sorted(df["round"].unique())
    print(f"\n  📊 Price history across {len(rounds)} snapshots (rounds: {list(rounds)})")

    # Drivers
    drivers = df[df["type"] == "driver"].pivot_table(
        index="name", columns="round", values="price", aggfunc="last"
    ).sort_values(by=rounds[-1], ascending=False)
    print("\n  DRIVERS:")
    header = f"    {'Name':15s}" + "".join(f"  R{r:<4}" for r in rounds)
    print(header)
    print("    " + "-" * len(header))
    for name, row in drivers.iterrows():
        vals = "".join(f"  {row.get(r, 0):5.1f}" for r in rounds)
        total_change = row.get(rounds[-1], 0) - row.get(rounds[0], 0)
        sign = "+" if total_change >= 0 else ""
        print(f"    {name:15s}{vals}  ({sign}{total_change:.1f})")

    # Constructors
    constrs = df[df["type"] == "constructor"].pivot_table(
        index="name", columns="round", values="price", aggfunc="last"
    ).sort_values(by=rounds[-1], ascending=False)
    print("\n  CONSTRUCTORS:")
    header = f"    {'Name':20s}" + "".join(f"  R{r:<4}" for r in rounds)
    print(header)
    print("    " + "-" * len(header))
    for name, row in constrs.iterrows():
        vals = "".join(f"  {row.get(r, 0):5.1f}" for r in rounds)
        total_change = row.get(rounds[-1], 0) - row.get(rounds[0], 0)
        sign = "+" if total_change >= 0 else ""
        print(f"    {name:20s}{vals}  ({sign}{total_change:.1f})")


# Load current prices at import time
DRIVER_PRICES, CONSTRUCTOR_PRICES = load_current_prices()

# Number of future rounds to look ahead for transfer planning
LOOKAHEAD_ROUNDS = 3
# Discount factor per round for future predictions (value today > value next week)
LOOKAHEAD_DISCOUNT = 0.7


# ──────────────────────────────────────────────────────────────
# Multi-Round Lookahead & Value Analysis
# ──────────────────────────────────────────────────────────────

def predict_lookahead(round_num, num_rounds=LOOKAHEAD_ROUNDS):
    """
    Predict driver and constructor scores for the next `num_rounds` rounds.
    Returns dict: {round: (driver_scores, constructor_scores)}.
    """
    forecasts = {}
    max_round = len(RACE_CALENDAR_2026)
    for r in range(round_num, min(round_num + num_rounds, max_round + 1)):
        ds = predict_driver_scores(r)
        cs = predict_constructor_scores(r)
        if ds and cs:
            forecasts[r] = (ds, cs)
    return forecasts


def compute_multi_round_value(forecasts, target_round):
    """
    Compute discounted multi-round expected points for each driver/constructor.
    Rounds closer to now are weighted more heavily.
    Returns (driver_multi, constructor_multi) dicts.
    """
    driver_multi = {}
    constructor_multi = {}
    for r, (ds, cs) in sorted(forecasts.items()):
        discount = LOOKAHEAD_DISCOUNT ** (r - target_round)
        for d, pts in ds.items():
            driver_multi[d] = driver_multi.get(d, 0) + pts * discount
        for c, pts in cs.items():
            constructor_multi[c] = constructor_multi.get(c, 0) + pts * discount
    return driver_multi, constructor_multi


def compute_value_per_million(scores, prices):
    """
    Compute points-per-$1m for each entity.
    Returns dict: {name: pts_per_million}.
    """
    vpm = {}
    for name, pts in scores.items():
        price = prices.get(name, 1.0)
        vpm[name] = pts / max(price, 0.1)
    return vpm


def pick_best_boost(drivers, driver_scores, driver_std=None):
    """
    Choose the best boost driver considering both expected points and upside.
    If std (variance) info is available, prefer high-upside drivers when
    their expected score is close to the top.
    """
    if not drivers:
        return None

    if driver_std:
        # Upside score = expected + 0.5 * std (favours volatile high-scorers)
        upside = {d: driver_scores.get(d, 0) + 0.5 * driver_std.get(d, 0) for d in drivers}
        return max(drivers, key=lambda d: upside[d])
    else:
        return max(drivers, key=lambda d: driver_scores.get(d, 0))


def get_driver_std_estimates(round_num):
    """
    Volatility (std of the weekend total) per driver.  Comes from the weekend
    simulation, which knows about DNF risk, grid sensitivity and sprint weekends;
    falls back to the spread of recent results if the simulator is unavailable.
    """
    try:
        return predict_driver_std(round_num)
    except Exception as exc:
        print(f"  ℹ Simulator volatility unavailable ({exc.__class__.__name__}); using recent-result spread.")

    from f1_ml_predictor import load_driver_model, build_combined_data
    artifacts = load_driver_model()
    if not artifacts:
        return {}

    combined_df, _ = build_combined_data(round_num)
    if combined_df is None or combined_df.empty:
        return {}

    std_map = {}
    for driver in GRID_2026:
        ddf = combined_df[combined_df["driver"] == driver]
        if len(ddf) >= 3:
            std_map[driver] = float(ddf["fantasy_points"].tail(5).std())
        else:
            std_map[driver] = 8.0  # default moderate volatility
    return std_map


def display_weekend_forecast(round_num, driver_name=None, top_n=None):
    """Print the full per-event forecast (qualifying / sprint / race) for a round."""
    from f1_weekend_model import format_forecast
    fc = get_weekend_forecast(round_num)
    print("\n" + format_forecast(fc, top_n=top_n))
    if driver_name:
        try:
            print(f"\n  Component breakdown - {driver_name}:")
            for event, comps in fc.breakdown(driver_name).items():
                items = "  ".join(f"{k}={v:+.1f}" for k, v in comps.items())
                print(f"    {event:7s} {items}")
        except KeyError:
            print(f"  (unknown driver/constructor '{driver_name}')")
    m = fc.meta
    print(f"\n  Model: qualifying noise scale {m['q_cal'][0]:.2f} / race {m['r_cal'][0]:.2f}, "
          f"team-mate correlation {m['q_cal'][1]:.2f}/{m['r_cal'][1]:.2f}; "
          f"calibrated on {m['calibration_weekends'][1]} past weekends.")


def display_team_risk(drivers, constructors, boost, round_num):
    """Distribution of the user's team score plus chip / boost values from the simulation."""
    try:
        fc = get_weekend_forecast(round_num)
        ch = fc.chip_analysis(drivers, constructors, boost)
    except Exception as exc:
        print(f"  ℹ Team-risk analysis unavailable ({exc.__class__.__name__}).")
        return
    lo, hi = ch["team_p10_p90"]
    print("\n" + "=" * 70)
    print("  🎲 YOUR CURRENT TEAM - SIMULATED OUTCOME RANGE")
    print("=" * 70)
    print(f"    Expected {ch['team_mean']:.1f} pts  (±{ch['team_sd']:.1f}),  80% range {lo:.0f} to {hi:.0f}")
    print("    Boost options (extra points from doubling the driver):")
    for d, v in sorted(ch["boost"].items(), key=lambda kv: -kv[1]["x2_extra"]):
        mark = "  ← current" if d == boost else ""
        print(f"      {d:15s} +{v['x2_extra']:5.1f} expected (3x: +{v['x3_extra']:5.1f}), "
              f"±{v['sd']:.0f}, {v['p_negative']:.0%} chance of a negative score{mark}")
    print(f"    No Negative chip would add ≈ {ch['no_negative_gain']:.1f} pts to this team this weekend")


def display_value_analysis(driver_scores, constructor_scores, round_num,
                           forecasts=None, driver_std=None):
    """Display points-per-$m value rankings and multi-round outlook."""
    print("\n" + "=" * 70)
    print("  📈 VALUE ANALYSIS")
    print("=" * 70)

    # Points per $m
    d_vpm = compute_value_per_million(driver_scores, DRIVER_PRICES)
    c_vpm = compute_value_per_million(constructor_scores, CONSTRUCTOR_PRICES)

    print("\n  💎 Best Value Drivers (pts/$m):")
    for d, vpm in sorted(d_vpm.items(), key=lambda x: -x[1])[:8]:
        pts = driver_scores.get(d, 0)
        price = DRIVER_PRICES.get(d, 0)
        std_str = ""
        if driver_std and d in driver_std:
            std_str = f"  ±{driver_std[d]:.0f}"
        print(f"    {d:15s}  {pts:5.1f} pts / ${price:4.1f}m = {vpm:4.2f} pts/$m{std_str}")

    print("\n  💎 Best Value Constructors (pts/$m):")
    for c, vpm in sorted(c_vpm.items(), key=lambda x: -x[1])[:6]:
        pts = constructor_scores.get(c, 0)
        price = CONSTRUCTOR_PRICES.get(c, 0)
        print(f"    {c:20s}  {pts:5.1f} pts / ${price:4.1f}m = {vpm:4.2f} pts/$m")

    # Multi-round outlook
    if forecasts and len(forecasts) > 1:
        print(f"\n  🔭 {LOOKAHEAD_ROUNDS}-ROUND OUTLOOK (Rounds {min(forecasts)}–{max(forecasts)}):")

        # Show upcoming sprint weekends
        upcoming_sprints = []
        for r in sorted(forecasts.keys()):
            race = get_race_info(r)
            if race and race["sprint"]:
                upcoming_sprints.append(f"R{r} {race['name']}")
        if upcoming_sprints:
            print(f"     Sprint weekends ahead: {', '.join(upcoming_sprints)}")

        d_multi, c_multi = compute_multi_round_value(forecasts, round_num)

        print("\n     Top Drivers (discounted multi-round value):")
        for d, val in sorted(d_multi.items(), key=lambda x: -x[1])[:8]:
            this_round = driver_scores.get(d, 0)
            print(f"       {d:15s}  R{round_num}={this_round:5.1f}  Multi={val:5.1f}")

        print("\n     Top Constructors (discounted multi-round value):")
        for c, val in sorted(c_multi.items(), key=lambda x: -x[1])[:6]:
            this_round = constructor_scores.get(c, 0)
            print(f"       {c:20s}  R{round_num}={this_round:5.1f}  Multi={val:5.1f}")


# ──────────────────────────────────────────────────────────────
# User Input Helpers
# ──────────────────────────────────────────────────────────────

def input_team():
    """Prompt user for current team details."""
    print("\n" + "=" * 60)
    print("  📋 ENTER YOUR CURRENT FANTASY F1 TEAM")
    print("=" * 60)

    print("\n  Available drivers:")
    for i, (name, price) in enumerate(sorted(DRIVER_PRICES.items(), key=lambda x: -x[1]), 1):
        team = GRID_2026.get(name, "?")
        print(f"    {i:2d}. {name:15s} (${price:.1f}m) - {team}")

    print("\n  Enter 5 driver names (comma-separated):")
    while True:
        raw = input("  → ").strip()
        drivers = [d.strip().title() for d in raw.split(",")]
        if len(drivers) != 5:
            print(f"  ⚠ Need exactly 5 drivers, got {len(drivers)}. Try again.")
            continue
        invalid = [d for d in drivers if d not in DRIVER_PRICES]
        if invalid:
            print(f"  ⚠ Unknown drivers: {invalid}. Try again.")
            continue
        break

    print("\n  Available constructors:")
    for i, (name, price) in enumerate(sorted(CONSTRUCTOR_PRICES.items(), key=lambda x: -x[1]), 1):
        print(f"    {i:2d}. {name:20s} (${price:.1f}m)")

    print("\n  Enter 2 constructor names (comma-separated):")
    while True:
        raw = input("  → ").strip()
        constructors = [c.strip() for c in raw.split(",")]
        # Try to match case-insensitively
        matched = []
        for c in constructors:
            found = False
            for k in CONSTRUCTOR_PRICES:
                if k.lower() == c.lower():
                    matched.append(k)
                    found = True
                    break
            if not found:
                matched.append(c)
        constructors = matched

        if len(constructors) != 2:
            print(f"  ⚠ Need exactly 2 constructors, got {len(constructors)}. Try again.")
            continue
        invalid = [c for c in constructors if c not in CONSTRUCTOR_PRICES]
        if invalid:
            print(f"  ⚠ Unknown constructors: {invalid}. Try again.")
            continue
        break

    print("\n  Who is your 2x boosted driver?")
    while True:
        boost = input("  → ").strip().title()
        if boost in drivers:
            break
        print(f"  ⚠ Must be one of your drivers: {drivers}")

    return drivers, constructors, boost


def input_race():
    """Prompt user for which race to optimize for."""
    print("\n  📅 2026 RACE CALENDAR:")
    for r in RACE_CALENDAR_2026:
        sprint_tag = " 🏎 Sprint" if r["sprint"] else ""
        print(f"    Round {r['round']:2d}: {r['name']:25s} ({r['location']}){sprint_tag}")

    print("\n  Enter the round number for the upcoming race:")
    while True:
        try:
            round_num = int(input("  → ").strip())
            race = get_race_info(round_num)
            if race:
                print(f"  ✓ Selected: {race['name']} ({race['location']})")
                return round_num
            print("  ⚠ Invalid round number.")
        except ValueError:
            print("  ⚠ Enter a number.")


def input_resources(current_drivers, current_constructors):
    """Prompt for budget, transfers, and chips."""
    print("\n  💰 RESOURCES:")

    # Calculate current team value
    team_value = (sum(DRIVER_PRICES.get(d, 0) for d in current_drivers) +
                  sum(CONSTRUCTOR_PRICES.get(c, 0) for c in current_constructors))
    print(f"  Current team value: ${team_value:.1f}m")

    spare = float(input("  Spare budget (money in the bank, $M): ").strip())
    budget = team_value + spare
    print(f"  ✓ Total budget: ${budget:.1f}m (team ${team_value:.1f}m + spare ${spare:.1f}m)")
    free_transfers = int(input("  Available free transfers (usually 1-3): ").strip())

    print("\n  🎫 AVAILABLE CHIPS (enter comma-separated, or 'none'):")
    print("    Options: wildcard, 3x_booster, no_negative, final_fix, autopilot")
    chips_raw = input("  → ").strip().lower()
    if chips_raw in ("none", ""):
        chips = []
    else:
        chips = [c.strip() for c in chips_raw.split(",")]

    return budget, free_transfers, chips


# ──────────────────────────────────────────────────────────────
# Optimizer Engine
# ──────────────────────────────────────────────────────────────

def calculate_team_score(drivers, constructors, boost_driver,
                          driver_scores, constructor_scores,
                          chip=None,
                          driver_price_changes=None,
                          constructor_price_changes=None,
                          multi_round_driver=None,
                          multi_round_constructor=None,
                          multi_round_weight=0.15):
    """
    Calculate predicted fantasy score for a team configuration.

    Returns (fantasy_pts, price_value, composite_score):
      - fantasy_pts:    raw predicted fantasy points for the weekend
      - price_value:    sum of expected price changes ($m) for team members
      - composite_score: fantasy_pts + price_value * PRICE_VALUE_WEIGHT
                         + multi-round lookahead bonus
    """
    d_pts = sum(driver_scores.get(d, 0) for d in drivers)

    # Boost: 2x on boosted driver (add their score again)
    boost_pts = driver_scores.get(boost_driver, 0)

    # 3x booster chip: triple instead of double
    if chip == "3x_booster":
        boost_pts *= 2  # Already counted once in d_pts, add 2x more
    # Normal: already counted once, add 1x more (= 2x total)

    c_pts = sum(constructor_scores.get(c, 0) for c in constructors)

    total = d_pts + boost_pts + c_pts

    # No negative chip: floor at 0 for each driver
    if chip == "no_negative":
        total = sum(max(0, driver_scores.get(d, 0)) for d in drivers)
        total += max(0, driver_scores.get(boost_driver, 0))  # boost portion
        total += sum(max(0, constructor_scores.get(c, 0)) for c in constructors)

    # Price-change value for the team
    pv = 0.0
    if driver_price_changes:
        pv += sum(driver_price_changes.get(d, 0.0) for d in drivers)
    if constructor_price_changes:
        pv += sum(constructor_price_changes.get(c, 0.0) for c in constructors)

    # Multi-round lookahead bonus (rewards picks that score well over next few rounds)
    mr_bonus = 0.0
    if multi_round_driver and multi_round_constructor:
        mr_bonus += sum(multi_round_driver.get(d, 0) for d in drivers)
        mr_bonus += sum(multi_round_constructor.get(c, 0) for c in constructors)
        mr_bonus *= multi_round_weight

    composite = total + pv * PRICE_VALUE_WEIGHT + mr_bonus
    return total, pv, composite


def find_optimal_transfers(current_drivers, current_constructors, current_boost,
                            driver_scores, constructor_scores,
                            budget, free_transfers, chips,
                            top_n=10,
                            driver_price_changes=None,
                            constructor_price_changes=None,
                            driver_std=None,
                            multi_round_driver=None,
                            multi_round_constructor=None):
    """
    Find optimal team changes considering:
    - Using 0, 1, or 2 transfers
    - Saving a transfer for next week (bank to 3)
    - Using a chip (wildcard = unlimited transfers)
    - Price-change optimisation (buy risers, sell fallers)
    - Multi-round lookahead (favour picks that score well over coming rounds)
    - Variance-aware boost selection (favour high-upside drivers for 2x)
    """
    all_drivers = sorted(DRIVER_PRICES.keys())
    all_constructors = sorted(CONSTRUCTOR_PRICES.keys())

    results = []
    pc_kw = dict(driver_price_changes=driver_price_changes,
                 constructor_price_changes=constructor_price_changes,
                 driver_std=driver_std,
                 multi_round_driver=multi_round_driver,
                 multi_round_constructor=multi_round_constructor)

    # Strategy 1: No changes (keep team, save transfer)
    # Recalculate boost for hold strategy too (variance-aware)
    hold_boost = pick_best_boost(current_drivers, driver_scores, driver_std)
    pts, pv, composite = calculate_team_score(
        current_drivers, current_constructors, hold_boost,
        driver_scores, constructor_scores,
        driver_price_changes=driver_price_changes,
        constructor_price_changes=constructor_price_changes,
        multi_round_driver=multi_round_driver,
        multi_round_constructor=multi_round_constructor,
    )
    results.append({
        "strategy": "HOLD (save transfer for next week)",
        "drivers": tuple(current_drivers),
        "constructors": tuple(current_constructors),
        "boost": current_boost,
        "trades": 0,
        "penalty": 0,
        "predicted_pts": pts,
        "price_value": pv,
        "composite_score": composite,
        "cost": sum(DRIVER_PRICES.get(d, 0) for d in current_drivers) +
                sum(CONSTRUCTOR_PRICES.get(c, 0) for c in current_constructors),
        "chip_used": None,
    })

    # Strategy 2: Use available free transfers (brute force combos)
    max_trades = free_transfers  # Don't exceed free to avoid penalty by default

    for num_trades in range(1, max_trades + 1):
        _search_trades(
            current_drivers, current_constructors, current_boost,
            driver_scores, constructor_scores, budget,
            num_trades, 0, f"USE {num_trades} FREE TRANSFER(S)",
            all_drivers, all_constructors, results, top_n, chip=None, **pc_kw
        )

    # Strategy 3: Use extra trade with penalty
    if free_transfers < 3:
        extra = free_transfers + 1
        penalty = TRADE_PENALTY_PTS
        _search_trades(
            current_drivers, current_constructors, current_boost,
            driver_scores, constructor_scores, budget,
            extra, penalty, f"USE {extra} TRADES ({extra - free_transfers} over = -{penalty} pts)",
            all_drivers, all_constructors, results, top_n, chip=None, **pc_kw
        )

    # Strategy 4: Chips
    if "wildcard" in chips:
        _search_all_teams(
            driver_scores, constructor_scores, budget,
            "WILDCARD (unlimited transfers)", all_drivers, all_constructors,
            results, top_n, chip="wildcard",
            current_drivers=current_drivers, current_constructors=current_constructors,
            **pc_kw
        )

    if "3x_booster" in chips:
        _search_trades(
            current_drivers, current_constructors, current_boost,
            driver_scores, constructor_scores, budget,
            free_transfers, 0, f"3x BOOSTER + {free_transfers} free transfer(s)",
            all_drivers, all_constructors, results, top_n, chip="3x_booster", **pc_kw
        )

    if "no_negative" in chips:
        _search_trades(
            current_drivers, current_constructors, current_boost,
            driver_scores, constructor_scores, budget,
            free_transfers, 0, f"NO NEGATIVE + {free_transfers} free transfer(s)",
            all_drivers, all_constructors, results, top_n, chip="no_negative", **pc_kw
        )

    # Deduplicate: keep best strategy per unique (drivers, constructors, boost) combo
    seen = set()
    deduped = []
    results.sort(key=lambda x: x["composite_score"], reverse=True)
    for r in results:
        key = (r["drivers"], r["constructors"], r["boost"])
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    return deduped[:top_n]


def _search_trades(current_drivers, current_constructors, current_boost,
                    driver_scores, constructor_scores, budget,
                    max_trades, penalty, strategy_name,
                    all_drivers, all_constructors, results, top_n, chip=None,
                    driver_price_changes=None, constructor_price_changes=None,
                    driver_std=None,
                    multi_round_driver=None, multi_round_constructor=None):
    """Search through trade combinations up to max_trades."""
    curr_d_set = set(current_drivers)
    curr_c_set = set(current_constructors)

    # Try all combinations of driver swaps (0..max_trades drivers, rest constructors)
    for d_out_count in range(min(max_trades, 5) + 1):
        c_out_count_max = max_trades - d_out_count
        if c_out_count_max > 2:
            c_out_count_max = 2

        for c_out_count in range(min(c_out_count_max, 2) + 1):
            if d_out_count + c_out_count == 0:
                continue
            if d_out_count + c_out_count > max_trades:
                continue

            # Choose which drivers to swap out
            for d_out in combinations(current_drivers, d_out_count):
                remaining_d = [d for d in current_drivers if d not in d_out]
                available_d = [d for d in all_drivers if d not in remaining_d]

                for d_in in combinations(available_d, d_out_count):
                    new_drivers = sorted(remaining_d + list(d_in))

                    for c_out in combinations(current_constructors, c_out_count):
                        remaining_c = [c for c in current_constructors if c not in c_out]
                        available_c = [c for c in all_constructors if c not in remaining_c]

                        for c_in in combinations(available_c, c_out_count):
                            new_constructors = sorted(remaining_c + list(c_in))

                            # Budget check
                            cost = (sum(DRIVER_PRICES.get(d, 0) for d in new_drivers) +
                                    sum(CONSTRUCTOR_PRICES.get(c, 0) for c in new_constructors))
                            if cost > budget:
                                continue

                            # Find best boost driver (variance-aware)
                            best_boost = pick_best_boost(new_drivers, driver_scores, driver_std)

                            pts, pv, composite = calculate_team_score(
                                new_drivers, new_constructors, best_boost,
                                driver_scores, constructor_scores, chip=chip,
                                driver_price_changes=driver_price_changes,
                                constructor_price_changes=constructor_price_changes,
                                multi_round_driver=multi_round_driver,
                                multi_round_constructor=multi_round_constructor,
                            )

                            results.append({
                                "strategy": strategy_name,
                                "drivers": tuple(new_drivers),
                                "constructors": tuple(new_constructors),
                                "boost": best_boost,
                                "trades": d_out_count + c_out_count,
                                "penalty": penalty,
                                "predicted_pts": pts - penalty,
                                "price_value": pv,
                                "composite_score": composite - penalty,
                                "cost": cost,
                                "chip_used": chip,
                            })


def _search_all_teams(driver_scores, constructor_scores, budget,
                       strategy_name, all_drivers, all_constructors,
                       results, top_n, chip=None,
                       current_drivers=None, current_constructors=None,
                       driver_price_changes=None, constructor_price_changes=None,
                       driver_std=None,
                       multi_round_driver=None, multi_round_constructor=None):
    """Wildcard: search all possible team combinations."""
    best = []
    for con_combo in combinations(all_constructors, 2):
        c_cost = sum(CONSTRUCTOR_PRICES[c] for c in con_combo)
        remaining = budget - c_cost
        if remaining < 0:
            continue

        for drv_combo in combinations(all_drivers, 5):
            d_cost = sum(DRIVER_PRICES[d] for d in drv_combo)
            if d_cost > remaining:
                continue

            best_boost = pick_best_boost(list(drv_combo), driver_scores, driver_std)
            pts, pv, composite = calculate_team_score(
                list(drv_combo), list(con_combo), best_boost,
                driver_scores, constructor_scores, chip=chip,
                driver_price_changes=driver_price_changes,
                constructor_price_changes=constructor_price_changes,
                multi_round_driver=multi_round_driver,
                multi_round_constructor=multi_round_constructor,
            )

            # Track trades
            trades = 0
            if current_drivers:
                trades += len(set(drv_combo) - set(current_drivers))
            if current_constructors:
                trades += len(set(con_combo) - set(current_constructors))

            entry = {
                "strategy": strategy_name,
                "drivers": drv_combo,
                "constructors": con_combo,
                "boost": best_boost,
                "trades": trades,
                "penalty": 0,
                "predicted_pts": pts,
                "price_value": pv,
                "composite_score": composite,
                "cost": d_cost + c_cost,
                "chip_used": chip,
            }

            if len(best) < top_n:
                best.append(entry)
                best.sort(key=lambda x: x["composite_score"], reverse=True)
            elif composite > best[-1]["composite_score"]:
                best[-1] = entry
                best.sort(key=lambda x: x["composite_score"], reverse=True)

    results.extend(best)


# ──────────────────────────────────────────────────────────────
# Results Tracking
# ──────────────────────────────────────────────────────────────

def record_actual_results():
    """After a race, record what actually happened for your team."""
    print("\n" + "=" * 60)
    print("  📝 RECORD ACTUAL RACE RESULTS")
    print("=" * 60)

    round_num = int(input("\n  Race round number: ").strip())
    race = get_race_info(round_num)
    race_name = race["name"] if race else f"Round {round_num}"

    print("  Enter your 5 drivers (comma-separated):")
    drivers = [d.strip().title() for d in input("  → ").strip().split(",")]

    print("  Enter your 2 constructors (comma-separated):")
    constructors_raw = input("  → ").strip().split(",")
    constructors = []
    for c in constructors_raw:
        c = c.strip()
        for k in CONSTRUCTOR_PRICES:
            if k.lower() == c.lower():
                constructors.append(k)
                break
        else:
            constructors.append(c)

    boost = input("  Who was your 2x driver? ").strip().title()
    chip = input("  Chip used (or 'none'): ").strip().lower()
    if chip == "none":
        chip = ""
    actual_pts = float(input("  Total fantasy points scored: ").strip())

    row = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "round": round_num,
        "race": race_name,
        "driver_1": drivers[0] if len(drivers) > 0 else "",
        "driver_2": drivers[1] if len(drivers) > 1 else "",
        "driver_3": drivers[2] if len(drivers) > 2 else "",
        "driver_4": drivers[3] if len(drivers) > 3 else "",
        "driver_5": drivers[4] if len(drivers) > 4 else "",
        "constructor_1": constructors[0] if len(constructors) > 0 else "",
        "constructor_2": constructors[1] if len(constructors) > 1 else "",
        "boost_2x": boost,
        "chip_used": chip,
        "actual_points": actual_pts,
    }

    file_exists = os.path.exists(HISTORY_FILE)
    with open(HISTORY_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"\n  ✅ Results saved to {HISTORY_FILE}")
    _show_season_summary()


def _show_season_summary():
    """Show running season totals from the history file."""
    if not os.path.exists(HISTORY_FILE):
        return

    df = pd.read_csv(HISTORY_FILE)
    if df.empty:
        return

    total = df["actual_points"].sum()
    races = len(df)
    avg = total / races if races > 0 else 0

    print(f"\n  📊 Season Summary: {races} races, {total:.0f} total pts, {avg:.1f} avg/race")


# ──────────────────────────────────────────────────────────────
# Display
# ──────────────────────────────────────────────────────────────

def display_recommendations(results, current_drivers, current_constructors, round_num,
                            driver_price_changes=None, constructor_price_changes=None):
    """Display optimization results with price-change insights."""
    race = get_race_info(round_num)
    race_name = race["name"] if race else f"Round {round_num}"
    has_price = driver_price_changes or constructor_price_changes

    print("\n" + "=" * 70)
    print(f"  🏎  OPTIMIZATION RESULTS - {race_name}")
    if has_price:
        print(f"  (Scoring: fantasy pts + price-change value × {PRICE_VALUE_WEIGHT})")
    print("=" * 70)

    for rank, r in enumerate(results[:10], 1):
        in_d = set(r["drivers"]) - set(current_drivers)
        out_d = set(current_drivers) - set(r["drivers"])
        in_c = set(r["constructors"]) - set(current_constructors)
        out_c = set(current_constructors) - set(r["constructors"])

        chip_str = f" | Chip: {r['chip_used']}" if r["chip_used"] else ""
        penalty_str = f" | Penalty: -{r['penalty']} pts" if r["penalty"] > 0 else ""

        pv = r.get("price_value", 0.0)
        pv_str = ""
        if has_price and abs(pv) > 0.001:
            arrow = "+" if pv > 0 else ""
            pv_str = f" | Price Δ: {arrow}${pv:.1f}m"

        print(f"\n  ┌─── #{rank}  {r['strategy']}")
        print(f"  │  Predicted: {r['predicted_pts']:.0f} pts{pv_str} | Cost: ${r['cost']:.1f}m | Trades: {r['trades']}{penalty_str}{chip_str}")
        if has_price:
            print(f"  │  Composite: {r.get('composite_score', r['predicted_pts']):.0f} (pts + price value)")
        print(f"  │  Boost 2x: {r['boost']}")

        if in_d or out_d or in_c or out_c:
            print("  │")
            if out_d:
                changes = []
                for d in out_d:
                    pc = driver_price_changes.get(d, 0.0) if driver_price_changes else 0.0
                    tag = f" (${pc:+.1f})" if has_price and abs(pc) > 0.001 else ""
                    changes.append(f"{d}{tag}")
                print(f"  │  🔴 OUT: {', '.join(changes)}")
            if in_d:
                changes = []
                for d in in_d:
                    pc = driver_price_changes.get(d, 0.0) if driver_price_changes else 0.0
                    tag = f" (${pc:+.1f})" if has_price and abs(pc) > 0.001 else ""
                    changes.append(f"{d}{tag}")
                print(f"  │  🟢 IN:  {', '.join(changes)}")
            if out_c:
                changes = []
                for c in out_c:
                    pc = constructor_price_changes.get(c, 0.0) if constructor_price_changes else 0.0
                    tag = f" (${pc:+.1f})" if has_price and abs(pc) > 0.001 else ""
                    changes.append(f"{c}{tag}")
                print(f"  │  🔴 Constructor OUT: {', '.join(changes)}")
            if in_c:
                changes = []
                for c in in_c:
                    pc = constructor_price_changes.get(c, 0.0) if constructor_price_changes else 0.0
                    tag = f" (${pc:+.1f})" if has_price and abs(pc) > 0.001 else ""
                    changes.append(f"{c}{tag}")
                print(f"  │  🟢 Constructor IN:  {', '.join(changes)}")
        else:
            print("  │  → No changes (keep current team)")

        print(f"  │  Team: {', '.join(r['drivers'])} + {', '.join(r['constructors'])}")
        print(f"  └{'─' * 67}")

    # Transfer strategy recommendation
    if len(results) > 1:
        best = results[0]
        hold = next((r for r in results if r["trades"] == 0), None)

        print("\n" + "=" * 70)
        print("  💡 RECOMMENDATION:")
        if hold and best["composite_score"] - hold["composite_score"] < 5:
            print("  → SAVE your transfers. The gain from trading is minimal (<5 composite pts).")
            print(f"    Holding: {hold['predicted_pts']:.0f} pts vs Best: {best['predicted_pts']:.0f} pts")
            if has_price and hold:
                print(f"    Price value — hold: ${hold.get('price_value', 0):.1f}m vs best: ${best.get('price_value', 0):.1f}m")
        elif best["chip_used"]:
            print(f"  → Use {best['chip_used'].upper()} chip! Significant advantage predicted.")
            hold_c = hold["composite_score"] if hold else 0
            print(f"    Expected composite gain: +{best['composite_score'] - hold_c:.0f} over holding")
        else:
            print(f"  → Make {best['trades']} trade(s): {best['strategy']}")
            if hold:
                print(f"    Expected gain: +{best['predicted_pts'] - hold['predicted_pts']:.0f} pts")
                if has_price:
                    pv_gain = best.get('price_value', 0) - hold.get('price_value', 0)
                    if abs(pv_gain) > 0.001:
                        print(f"    Price value gain: ${pv_gain:+.1f}m (team value change)")
        print("=" * 70)


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def _use_live_results(round_num):
    """If qualifying (or sprint qualifying) is already done, condition forecasts on the real grid."""
    known = fetch_known_results(round_num)
    register_known_results(round_num, known)
    if known:
        parts = ", ".join({"quali": "qualifying", "sprint_quali": "sprint qualifying"}[k] for k in known)
        print(f"  ✅ Using actual {parts} results - race forecast is conditioned on the real grid.")
    return known


def main():
    print("\n" + "=" * 60)
    print("  🏆 F1 FANTASY 2026 TEAM OPTIMIZER")
    print("=" * 60)
    print(f"  (Using prices as of Round {get_price_round()})")

    print("\n  What would you like to do?")
    print("    1. Optimize team for upcoming race")
    print("    2. Record actual race results")
    print("    3. View season history")
    print("    4. Update prices (scrape from f1fantasytools.com or manual)")
    print("    5. View price history")
    print("    6. Weekend forecast - every event (quali / sprint / race) with uncertainty")

    choice = input("\n  → ").strip()

    if choice == "6":
        round_num = input_race()
        _use_live_results(round_num)
        name = input("  Driver or constructor for a points breakdown (blank to skip): ").strip() or None
        display_weekend_forecast(round_num, driver_name=name)
        return

    if choice == "2":
        record_actual_results()
        return

    if choice == "3":
        _show_season_summary()
        if os.path.exists(HISTORY_FILE):
            df = pd.read_csv(HISTORY_FILE)
            print(df.to_string(index=False))
        else:
            print("  No history yet.")
        return

    if choice == "4":
        update_prices()
        # Reload prices globally
        global DRIVER_PRICES, CONSTRUCTOR_PRICES
        DRIVER_PRICES, CONSTRUCTOR_PRICES = load_current_prices()
        return

    if choice == "5":
        show_price_history()
        return

    # Choice 1: Optimize
    drivers, constructors, boost = input_team()
    round_num = input_race()
    budget, free_transfers, chips = input_resources(drivers, constructors)

    race = get_race_info(round_num)
    print(f"\n  🔮 Generating predictions for {race['name']}...")
    _use_live_results(round_num)

    driver_scores = predict_driver_scores(round_num)
    constructor_scores = predict_constructor_scores(round_num)

    if not driver_scores:
        print("  ⚠ Could not generate predictions. Make sure models are trained (run f1_trainer.py first).")
        return

    # Fetch price-change estimates from f1fantasytools.com thresholds
    print("\n  📈 Fetching price-change thresholds from f1fantasytools.com...")
    d_pc, c_pc = fetch_price_change_estimates(driver_scores, constructor_scores)
    has_price = bool(d_pc or c_pc)
    if has_price:
        print("  ✅ Price-change estimates loaded — optimizer will factor in price movements.")
    else:
        print("  ⚠ Could not fetch price thresholds — optimizing on points only.")
        d_pc, c_pc = None, None

    # Get driver volatility estimates for variance-aware boost selection
    print("  📊 Computing driver volatility estimates...")
    driver_std = get_driver_std_estimates(round_num)

    # Multi-round lookahead
    print(f"  🔭 Predicting next {LOOKAHEAD_ROUNDS} rounds for transfer planning...")
    forecasts = predict_lookahead(round_num, LOOKAHEAD_ROUNDS)
    d_multi, c_multi = None, None
    if len(forecasts) > 1:
        d_multi, c_multi = compute_multi_round_value(forecasts, round_num)
        print(f"  ✅ Multi-round outlook loaded (Rounds {min(forecasts)}–{max(forecasts)})")
    else:
        print("  ⚠ Only 1 round available — skipping multi-round lookahead.")

    print("\n  📊 Predicted Driver Scores:")
    for d, pts in sorted(driver_scores.items(), key=lambda x: -x[1]):
        marker = " ← YOUR TEAM" if d in drivers else ""
        pc_str = ""
        if d_pc and d in d_pc:
            pc = d_pc[d]
            if abs(pc) > 0.001:
                pc_str = f"  (price ${pc:+.1f}m)"
        std_str = f"  ±{driver_std.get(d, 0):.0f}" if driver_std.get(d) else ""
        print(f"    {d:15s} → {pts:6.1f} pts{std_str}{pc_str}{marker}")

    print("\n  📊 Predicted Constructor Scores:")
    for c, pts in sorted(constructor_scores.items(), key=lambda x: -x[1]):
        marker = " ← YOUR TEAM" if c in constructors else ""
        pc_str = ""
        if c_pc and c in c_pc:
            pc = c_pc[c]
            if abs(pc) > 0.001:
                pc_str = f"  (price ${pc:+.1f}m)"
        print(f"    {c:20s} → {pts:6.1f} pts{pc_str}{marker}")

    # Value analysis display
    display_value_analysis(driver_scores, constructor_scores, round_num,
                           forecasts=forecasts, driver_std=driver_std)

    print(f"\n  ⚙ Optimizing with {free_transfers} free transfers, budget ${budget:.1f}m...")
    if has_price:
        print(f"     (price-change weight: {PRICE_VALUE_WEIGHT}× per $1m)")
    if d_multi:
        print(f"     (multi-round lookahead: {LOOKAHEAD_ROUNDS} rounds, discount={LOOKAHEAD_DISCOUNT})")
    if driver_std:
        print(f"     (variance-aware boost selection enabled)")
    start = time.time()

    results = find_optimal_transfers(
        drivers, constructors, boost,
        driver_scores, constructor_scores,
        budget, free_transfers, chips,
        driver_price_changes=d_pc,
        constructor_price_changes=c_pc,
        driver_std=driver_std,
        multi_round_driver=d_multi,
        multi_round_constructor=c_multi,
    )

    elapsed = time.time() - start
    print(f"  Optimization complete in {elapsed:.1f}s ({len(results)} strategies evaluated)")

    display_recommendations(results, drivers, constructors, round_num,
                            driver_price_changes=d_pc,
                            constructor_price_changes=c_pc)
    display_team_risk(drivers, constructors, boost, round_num)


if __name__ == "__main__":
    main()
