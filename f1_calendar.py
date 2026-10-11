#!/usr/bin/env python3
"""
Calendar, grid and track profiles (f1_calendar.py)
==================================================
Light-weight (no sklearn / xgboost / network) home for:
  * the 2026 grid and race calendar
  * historical calendars used to order past races chronologically
  * name canonicalisation (accents, team renames)
  * per-circuit characteristics used as priors by the weekend simulator
"""

import unicodedata

# ──────────────────────────────────────────────────────────────
# 2026 Grid - Driver/Team Mapping
# ──────────────────────────────────────────────────────────────

GRID_2026 = {
    "Verstappen": "Red Bull Racing",
    "Hadjar": "Red Bull Racing",
    "Russell": "Mercedes",
    "Antonelli": "Mercedes",
    "Norris": "McLaren",
    "Piastri": "McLaren",
    "Leclerc": "Ferrari",
    "Hamilton": "Ferrari",
    "Gasly": "Alpine",
    "Colapinto": "Alpine",
    "Sainz": "Williams",
    "Albon": "Williams",
    "Alonso": "Aston Martin",
    "Stroll": "Aston Martin",
    "Bearman": "Haas F1 Team",
    "Ocon": "Haas F1 Team",
    "Hulkenberg": "Audi",
    "Bortoleto": "Audi",
    "Lawson": "Racing Bulls",
    "Lindblad": "Racing Bulls",
    "Perez": "Cadillac",
    "Bottas": "Cadillac",
}

# ──────────────────────────────────────────────────────────────
# 2026 Race Calendar (location names matching OpenF1 conventions)
# ──────────────────────────────────────────────────────────────

RACE_CALENDAR_2026 = [
    {"round": 1, "location": "Melbourne", "name": "Australian GP", "sprint": False},
    {"round": 2, "location": "Shanghai", "name": "Chinese GP", "sprint": True},
    {"round": 3, "location": "Suzuka", "name": "Japanese GP", "sprint": False},
    {"round": 4, "location": "Sakhir", "name": "Bahrain GP", "sprint": False},
    {"round": 5, "location": "Jeddah", "name": "Saudi Arabian GP", "sprint": False},
    {"round": 6, "location": "Miami", "name": "Miami GP", "sprint": True},
    {"round": 7, "location": "Montréal", "name": "Canadian GP", "sprint": True},
    {"round": 8, "location": "Monaco", "name": "Monaco GP", "sprint": False},
    {"round": 9, "location": "Barcelona", "name": "Barcelona-Catalunya GP", "sprint": False},
    {"round": 10, "location": "Spielberg", "name": "Austrian GP", "sprint": False},
    {"round": 11, "location": "Silverstone", "name": "British GP", "sprint": True},
    {"round": 12, "location": "Spa-Francorchamps", "name": "Belgian GP", "sprint": False},
    {"round": 13, "location": "Budapest", "name": "Hungarian GP", "sprint": False},
    {"round": 14, "location": "Zandvoort", "name": "Dutch GP", "sprint": True},
    {"round": 15, "location": "Monza", "name": "Italian GP", "sprint": False},
    {"round": 16, "location": "Madrid", "name": "Spanish GP", "sprint": False},
    {"round": 17, "location": "Baku", "name": "Azerbaijan GP", "sprint": False},
    {"round": 18, "location": "Marina Bay", "name": "Singapore GP", "sprint": True},
    {"round": 19, "location": "Austin", "name": "US GP", "sprint": False},
    {"round": 20, "location": "Mexico City", "name": "Mexico City GP", "sprint": False},
    {"round": 21, "location": "São Paulo", "name": "São Paulo GP", "sprint": False},
    {"round": 22, "location": "Las Vegas", "name": "Las Vegas GP", "sprint": False},
    {"round": 23, "location": "Lusail", "name": "Qatar GP", "sprint": False},
    {"round": 24, "location": "Yas Island", "name": "Abu Dhabi GP", "sprint": False},
]


def get_race_info(round_num):
    """Get race info from the calendar by round number."""
    for race in RACE_CALENDAR_2026:
        if race["round"] == round_num:
            return race
    return None


# Aliases for OpenF1 location names that differ from our calendar
LOCATION_ALIASES = {
    "monte carlo": "Monaco",
    "yas marina": "Yas Island",
    "miami gardens": "Miami",
    "imola": "Barcelona",        # Imola no longer on 2026 calendar, closest match
    "le castellet": "Barcelona",
    "montreal": "Montréal",
}


def round_for_location(location):
    """Map a location string to its 2026 round number (fuzzy + alias match)."""
    loc_lower = location.lower().strip()

    for alias, canonical in LOCATION_ALIASES.items():
        if alias in loc_lower:
            for race in RACE_CALENDAR_2026:
                if race["location"] == canonical:
                    return race["round"]

    for race in RACE_CALENDAR_2026:
        cal = race["location"].lower()
        if cal in loc_lower or loc_lower in cal:
            return race["round"]
    return None


def canon_location(location):
    """Calendar-style location name for an OpenF1 location (itself if unknown)."""
    rnd = round_for_location(location) if location else None
    if rnd is not None:
        info = get_race_info(rnd)
        # Imola/Le Castellet alias onto Barcelona only to find a round; keep the
        # real name so historical tracks stay distinct.
        if info and location.lower().strip() not in ("imola", "le castellet"):
            return info["location"]
    return location


# ──────────────────────────────────────────────────────────────
# Name canonicalisation
# ──────────────────────────────────────────────────────────────

def _fold(s):
    """Lower-case, accent-free version of a string."""
    return "".join(
        c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c)
    ).lower().strip()


_DRIVER_LOOKUP = {_fold(k): k for k in GRID_2026}
_DRIVER_ALIASES = {
    "hulkenberg": "Hulkenberg",
    "perez": "Perez",
    "kimi antonelli": "Antonelli",
    "andrea kimi antonelli": "Antonelli",
    "verstappen": "Verstappen",
}


def canon_driver(name):
    """Map an OpenF1 last_name (accents etc.) onto the GRID_2026 spelling when known."""
    f = _fold(name)
    if f in _DRIVER_LOOKUP:
        return _DRIVER_LOOKUP[f]
    if f in _DRIVER_ALIASES:
        return _DRIVER_ALIASES[f]
    # last token fallback ("Andrea Kimi Antonelli" -> "antonelli")
    last = f.split()[-1] if f else f
    return _DRIVER_LOOKUP.get(last, str(name))


_TEAM_ALIASES = {
    "kick sauber": "Audi",
    "sauber": "Audi",
    "alfa romeo": "Audi",
    "stake f1 team kick sauber": "Audi",
    "rb": "Racing Bulls",
    "visa cash app rb": "Racing Bulls",
    "rb f1 team": "Racing Bulls",
    "alphatauri": "Racing Bulls",
    "scuderia alphatauri": "Racing Bulls",
    "haas": "Haas F1 Team",
    "haas f1 team": "Haas F1 Team",
    "red bull": "Red Bull Racing",
    "oracle red bull racing": "Red Bull Racing",
    "mercedes-amg petronas": "Mercedes",
    "alpine f1 team": "Alpine",
    "aston martin aramco": "Aston Martin",
    "scuderia ferrari": "Ferrari",
    "mclaren f1 team": "McLaren",
    "williams racing": "Williams",
    "cadillac f1 team": "Cadillac",
}
_TEAM_LOOKUP = {_fold(t): t for t in set(GRID_2026.values())}


def canon_team(name):
    """Map OpenF1 team_name onto the 2026 constructor name (Sauber->Audi, RB->Racing Bulls...)."""
    f = _fold(name)
    if f in _TEAM_LOOKUP:
        return _TEAM_LOOKUP[f]
    return _TEAM_ALIASES.get(f, str(name))


# ──────────────────────────────────────────────────────────────
# Historical calendars (to order past races chronologically)
# ──────────────────────────────────────────────────────────────

HISTORICAL_CALENDARS = {
    2024: ["Sakhir", "Jeddah", "Melbourne", "Suzuka", "Shanghai", "Miami", "Imola",
           "Monaco", "Montréal", "Barcelona", "Spielberg", "Silverstone", "Budapest",
           "Spa-Francorchamps", "Zandvoort", "Monza", "Baku", "Marina Bay", "Austin",
           "Mexico City", "São Paulo", "Las Vegas", "Lusail", "Yas Island"],
    2025: ["Melbourne", "Shanghai", "Suzuka", "Sakhir", "Jeddah", "Miami", "Imola",
           "Monaco", "Barcelona", "Montréal", "Spielberg", "Silverstone",
           "Spa-Francorchamps", "Budapest", "Zandvoort", "Monza", "Baku", "Marina Bay",
           "Austin", "Mexico City", "São Paulo", "Las Vegas", "Lusail", "Yas Island"],
    2026: [r["location"] for r in RACE_CALENDAR_2026],
}


def historical_round(year, location):
    """Round number for a past race, or None if unknown. Accent/alias tolerant."""
    cal = HISTORICAL_CALENDARS.get(year)
    if not cal:
        return None
    target = _fold(location)
    for i, loc in enumerate(cal, 1):
        f = _fold(loc)
        if f == target or f in target or target in f:
            return i
    for alias, canon in LOCATION_ALIASES.items():
        if alias in target and canon in cal:
            return cal.index(canon) + 1
    return None


# ──────────────────────────────────────────────────────────────
# Track profiles (informed priors - edit freely)
# ──────────────────────────────────────────────────────────────
#   stickiness : how strongly grid order carries into the race (1.0 = average;
#                Monaco ~2, Baku / Spa / Bahrain < 1).
#   chaos      : multiplier on DNF probability / race variance (safety cars, walls).
#   overtaking : multiplier on on-track overtake count.
#   stops      : mean pit stops per car in a dry race.

_DEFAULT_PROFILE = {"stickiness": 1.0, "chaos": 1.0, "overtaking": 1.0, "stops": 1.9}

TRACK_PROFILES = {
    "Melbourne":         {"stickiness": 1.00, "chaos": 1.25, "overtaking": 1.00, "stops": 1.6},
    "Shanghai":          {"stickiness": 0.85, "chaos": 1.00, "overtaking": 1.30, "stops": 2.0},
    "Suzuka":            {"stickiness": 1.20, "chaos": 0.95, "overtaking": 0.80, "stops": 1.9},
    "Sakhir":            {"stickiness": 0.80, "chaos": 1.00, "overtaking": 1.30, "stops": 2.2},
    "Jeddah":            {"stickiness": 1.00, "chaos": 1.50, "overtaking": 1.00, "stops": 1.5},
    "Miami":             {"stickiness": 0.95, "chaos": 1.10, "overtaking": 1.00, "stops": 1.7},
    "Montréal":          {"stickiness": 0.95, "chaos": 1.30, "overtaking": 1.00, "stops": 1.7},
    "Monaco":            {"stickiness": 2.00, "chaos": 1.10, "overtaking": 0.35, "stops": 1.2},
    "Barcelona":         {"stickiness": 1.30, "chaos": 0.80, "overtaking": 0.80, "stops": 2.1},
    "Spielberg":         {"stickiness": 0.90, "chaos": 1.00, "overtaking": 1.10, "stops": 2.0},
    "Silverstone":       {"stickiness": 0.95, "chaos": 0.95, "overtaking": 1.10, "stops": 1.9},
    "Spa-Francorchamps": {"stickiness": 0.80, "chaos": 1.20, "overtaking": 1.30, "stops": 2.0},
    "Budapest":          {"stickiness": 1.40, "chaos": 0.90, "overtaking": 0.70, "stops": 2.0},
    "Zandvoort":         {"stickiness": 1.50, "chaos": 1.00, "overtaking": 0.60, "stops": 2.0},
    "Monza":             {"stickiness": 0.90, "chaos": 1.00, "overtaking": 1.00, "stops": 1.3},
    "Madrid":            {"stickiness": 1.10, "chaos": 1.10, "overtaking": 0.90, "stops": 1.9},
    "Baku":              {"stickiness": 0.80, "chaos": 1.50, "overtaking": 1.30, "stops": 1.6},
    "Marina Bay":        {"stickiness": 1.30, "chaos": 1.50, "overtaking": 0.80, "stops": 2.0},
    "Austin":            {"stickiness": 0.90, "chaos": 1.10, "overtaking": 1.20, "stops": 2.0},
    "Mexico City":       {"stickiness": 1.10, "chaos": 1.00, "overtaking": 0.90, "stops": 1.6},
    "São Paulo":         {"stickiness": 0.80, "chaos": 1.30, "overtaking": 1.30, "stops": 2.0},
    "Las Vegas":         {"stickiness": 0.80, "chaos": 1.10, "overtaking": 1.30, "stops": 1.4},
    "Lusail":            {"stickiness": 1.10, "chaos": 1.00, "overtaking": 0.80, "stops": 2.4},
    "Yas Island":        {"stickiness": 1.10, "chaos": 0.90, "overtaking": 0.90, "stops": 1.5},
}


def track_profile(location):
    """Profile dict for a location (average profile if unknown)."""
    loc = canon_location(location) if location else location
    return dict(TRACK_PROFILES.get(loc, _DEFAULT_PROFILE))


# ──────────────────────────────────────────────────────────────
# Pre-season fantasy prices ($m)
# ──────────────────────────────────────────────────────────────

DEFAULT_DRIVER_PRICES = {
    "Verstappen": 27.7, "Russell": 27.4, "Norris": 27.2, "Piastri": 25.5,
    "Antonelli": 23.2, "Leclerc": 22.8, "Hamilton": 22.5, "Hadjar": 15.1,
    "Gasly": 12.0, "Sainz": 11.8, "Albon": 11.6, "Alonso": 10.0,
    "Stroll": 8.0, "Bearman": 7.4, "Ocon": 7.3, "Hulkenberg": 6.8,
    "Lawson": 6.5, "Bortoleto": 6.4, "Lindblad": 6.2, "Colapinto": 6.2,
    "Perez": 6.0, "Bottas": 5.9,
}

DEFAULT_CONSTRUCTOR_PRICES = {
    "Mercedes": 29.3, "McLaren": 28.7, "Red Bull Racing": 28.2,
    "Ferrari": 23.3, "Alpine": 12.5, "Williams": 12.0,
    "Aston Martin": 9.1, "Haas F1 Team": 7.4, "Audi": 5.4,
    "Racing Bulls": 6.3, "Cadillac": 4.8,
}
