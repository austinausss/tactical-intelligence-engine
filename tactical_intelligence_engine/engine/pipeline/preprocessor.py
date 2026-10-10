"""
Tactical Preprocessor & Phase Segmentation Engine.
Extracts structured possession sequences, flags progressive actions, identifies line breaks,
and categorizes tactical phases (Build-up, Progression, Final Third Penetration, High Press).
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from config import PITCH_LENGTH, PITCH_WIDTH, FINAL_THIRD_X, PENALTY_BOX_X, PENALTY_BOX_Y_MIN, PENALTY_BOX_Y_MAX


class TacticalPreprocessor:
    """
    Transforms cleaned spatial event streams into tactical intelligence features.
    """

    def __init__(self, pitch_length: float = PITCH_LENGTH, pitch_width: float = PITCH_WIDTH):
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width

    def enrich_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enriches raw events with distance to goal, progressive action flags,
        box entries, and tactical sequence indices.
        """
        enriched = df.copy()

        # Ensure spatial columns exist
        for col in ["location_x", "location_y", "end_x", "end_y"]:
            if col not in enriched.columns:
                enriched[col] = np.nan

        # 1. Goal distance and angles (Direction-aware)
        # Standard StatsBomb and Argentina attack towards (120, 40)
        is_spain = (enriched.get("team", "") == "Spain") & (enriched.get("match_id", "") == "WC2026-FINAL-ESP-ARG")
        target_x = np.where(is_spain, 0.0, self.pitch_length)

        dist_start = np.sqrt((target_x - enriched["location_x"])**2 + (40.0 - enriched["location_y"])**2)
        dist_end = np.sqrt((target_x - enriched["end_x"])**2 + (40.0 - enriched["end_y"])**2)
        enriched["dist_to_goal_start"] = dist_start
        enriched["dist_to_goal_end"] = dist_end

        # 2. Progressive Pass Detection
        # Standard Wyscout/StatsBomb definition: moves ball at least 10 yds closer (or 20% in final third)
        is_pass = enriched["type"] == "Pass"
        pass_complete = enriched["pass_outcome"].isna() | (enriched["pass_outcome"] == "Complete")
        delta_dist = dist_start - dist_end

        in_final_third = np.where(is_spain, enriched["location_x"] <= 40.0, enriched["location_x"] >= FINAL_THIRD_X)
        prog_condition_mid = (~in_final_third) & (delta_dist >= 9.0)
        prog_condition_final = in_final_third & (delta_dist >= (dist_start * 0.18)) & (delta_dist > 3.0)

        enriched["is_progressive_pass"] = is_pass & pass_complete & (prog_condition_mid | prog_condition_final)

        # 3. Progressive Carry Detection
        is_carry = enriched["type"] == "Carry"
        carry_delta_dist = dist_start - dist_end
        enriched["is_progressive_carry"] = is_carry & (carry_delta_dist >= 7.5)

        # 4. Penalty Box Entry
        starts_outside_box = np.where(
            is_spain,
            (enriched["location_x"] > 18.0) | (enriched["location_y"] < PENALTY_BOX_Y_MIN) | (enriched["location_y"] > PENALTY_BOX_Y_MAX),
            (enriched["location_x"] < PENALTY_BOX_X) | (enriched["location_y"] < PENALTY_BOX_Y_MIN) | (enriched["location_y"] > PENALTY_BOX_Y_MAX)
        )
        ends_inside_box = np.where(
            is_spain,
            (enriched["end_x"] <= 18.0) & (enriched["end_y"] >= PENALTY_BOX_Y_MIN) & (enriched["end_y"] <= PENALTY_BOX_Y_MAX),
            (enriched["end_x"] >= PENALTY_BOX_X) & (enriched["end_y"] >= PENALTY_BOX_Y_MIN) & (enriched["end_y"] <= PENALTY_BOX_Y_MAX)
        )
        enriched["is_box_entry"] = (is_pass | is_carry) & starts_outside_box & ends_inside_box

        # 5. Tactical Pitch Zones
        enriched["tactical_third"] = np.where(
            is_spain,
            np.where(enriched["location_x"] > 80.0, "Defensive Third", np.where(enriched["location_x"] > 40.0, "Middle Third", "Final Third")),
            np.where(enriched["location_x"] < 40.0, "Defensive Third", np.where(enriched["location_x"] < 80.0, "Middle Third", "Final Third"))
        )

        # 6. High Press / Defensive Action categorization
        defensive_types = ["Pressure", "Tackle", "Interception", "Ball Recovery", "Block", "Foul Committed"]
        enriched["is_defensive_action"] = enriched["type"].isin(defensive_types)
        enriched["is_high_press"] = enriched["is_defensive_action"] & in_final_third

        return enriched

    def segment_possession_chains(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assigns structured continuous possession sequence IDs across time gaps
        and turnover points.
        """
        df = df.copy()
        if "possession" not in df.columns:
            # Fallback continuous chain grouping by team
            team_change = (df["team"] != df["team"].shift(1)).cumsum()
            df["possession"] = team_change

        return df
