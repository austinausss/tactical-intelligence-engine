"""
Configuration and Tactical Constants for Football Tactical Intelligence Engine.
Coordinates: StatsBomb 120x80 yard coordinate system.
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Pitch Dimensions (StatsBomb standard in yards)
PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0
GOAL_WIDTH = 8.0
GOAL_Y_MIN = 36.0
GOAL_Y_MAX = 44.0
GOAL_CENTER = (120.0, 40.0)

# Tactical Zones
DEFENSIVE_THIRD_X = 40.0
MIDDLE_THIRD_X = 80.0
FINAL_THIRD_X = 80.0  # x >= 80 is final third
PENALTY_BOX_X = 102.0
PENALTY_BOX_Y_MIN = 18.0
PENALTY_BOX_Y_MAX = 62.0
SIX_YARD_BOX_X = 114.0
SIX_YARD_BOX_Y_MIN = 30.0
SIX_YARD_BOX_Y_MAX = 50.0

# Visual Styling (High-End Dark Tactical Theme)
STYLE_CONFIG = {
    "bg_color": "#0b0f19",          # Deep midnight obsidian
    "pitch_color": "#111827",       # Pitch dark surface
    "pitch_lines": "#374151",       # Subtle field markings
    "grass_stripe": "#0f172a",      # Subtle turf alternation
    "text_color": "#f3f4f6",        # Crisp white/silver text
    "text_muted": "#9ca3af",        # Muted slate text
    "accent_primary": "#00f2fe",    # Cyan / Electric blue
    "accent_secondary": "#4facfe",  # Royal bright blue
    
    # Team Colors: Argentina (Albiceleste) & Spain (La Roja)
    "argentina_primary": "#75AADB", # Sky blue
    "argentina_secondary": "#FFFFFF", # Sun white
    "argentina_accent": "#F6B40E",  # Sol de Mayo gold
    
    "spain_primary": "#DC2626",     # Spanish Red
    "spain_secondary": "#FBBF24",   # Spanish Gold
    "spain_accent": "#1E3A8A",      # Royal Navy
    
    # Outcome Colors
    "goal": "#10B981",              # Emerald green
    "saved": "#38BDF8",             # Sky blue
    "off_target": "#F43F5E",        # Rose / Crimson
    "blocked": "#6B7280",           # Slate gray
    "post": "#F59E0B",              # Amber
}

# 2026 World Cup Final Match Metadata (OneFootball / FIFA Verified)
FINAL_2026_METADATA = {
    "match_id": "WC2026-FINAL-ESP-ARG",
    "competition": "FIFA World Cup 2026",
    "stage": "Final",
    "venue": "MetLife Stadium, East Rutherford, NJ",
    "date": "2026-07-19",
    "attendance": 80663,
    "referee": "Slavko Vincic (Slovenia)",
    "home_team": "Spain",
    "away_team": "Argentina",
    "home_score": 1,
    "away_score": 0,
    "result_detail": "Spain 1 - 0 Argentina (AET)",
    "winning_goal": "Ferran Torres (106')",
    "red_card": "Enzo Fernandez (82' / 93' Red Card)",
    "home_manager": "Luis de la Fuente",
    "away_manager": "Lionel Scaloni",
    "home_formation": "4-3-2-1 Christmas Tree",
    "away_formation": "4-4-2 Dynamic Shape",
    "man_of_the_match": "Ferran Torres (Spain)",
    "golden_glove": "Unai Simon (Spain)",
    "dibu_saves_record": "Emiliano Martinez: 11 Saves (World Cup Final Record)",
}
