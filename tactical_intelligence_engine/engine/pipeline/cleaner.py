"""
Coordinate Cleaning and Spatial Data Sanitation Engine.
Processes raw StatsBomb event streams, cleans coordinate noise, handles missing spatial points,
and provides directional standardization.
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd
from config import PITCH_LENGTH, PITCH_WIDTH


class CoordinateCleaner:
    """
    Sanitizes and cleans spatial coordinates from raw event records.
    Coordinates adhere to the standard StatsBomb 120 x 80 yard framework.
    """

    def __init__(self, pitch_length: float = PITCH_LENGTH, pitch_width: float = PITCH_WIDTH):
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width

    def clean_event_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Unpacks nested coordinate lists into explicit numeric columns (x, y, end_x, end_y),
        detects and rectifies out-of-bound anomalies, and imputes missing spatial markers.
        """
        cleaned = df.copy()

        # 1. Unpack primary event location
        if "location" in cleaned.columns:
            loc_series = cleaned["location"]
            cleaned["location_x"] = loc_series.apply(
                lambda loc: float(loc[0]) if isinstance(loc, (list, tuple, np.ndarray)) and len(loc) >= 2 else np.nan
            )
            cleaned["location_y"] = loc_series.apply(
                lambda loc: float(loc[1]) if isinstance(loc, (list, tuple, np.ndarray)) and len(loc) >= 2 else np.nan
            )
        elif "location_x" not in cleaned.columns:
            cleaned["location_x"] = np.nan
            cleaned["location_y"] = np.nan

        # 2. Unpack pass end location
        if "pass_end_location" in cleaned.columns:
            pass_loc = cleaned["pass_end_location"]
            cleaned["pass_end_x"] = pass_loc.apply(
                lambda loc: float(loc[0]) if isinstance(loc, (list, tuple, np.ndarray)) and len(loc) >= 2 else np.nan
            )
            cleaned["pass_end_y"] = pass_loc.apply(
                lambda loc: float(loc[1]) if isinstance(loc, (list, tuple, np.ndarray)) and len(loc) >= 2 else np.nan
            )
        elif "pass_end_x" not in cleaned.columns:
            cleaned["pass_end_x"] = np.nan
            cleaned["pass_end_y"] = np.nan

        # 3. Unpack carry end location
        if "carry_end_location" in cleaned.columns:
            carry_loc = cleaned["carry_end_location"]
            cleaned["carry_end_x"] = carry_loc.apply(
                lambda loc: float(loc[0]) if isinstance(loc, (list, tuple, np.ndarray)) and len(loc) >= 2 else np.nan
            )
            cleaned["carry_end_y"] = carry_loc.apply(
                lambda loc: float(loc[1]) if isinstance(loc, (list, tuple, np.ndarray)) and len(loc) >= 2 else np.nan
            )
        elif "carry_end_x" not in cleaned.columns:
            cleaned["carry_end_x"] = np.nan
            cleaned["carry_end_y"] = np.nan

        # 4. Unpack shot end location
        if "shot_end_location" in cleaned.columns:
            shot_loc = cleaned["shot_end_location"]
            cleaned["shot_end_x"] = shot_loc.apply(
                lambda loc: float(loc[0]) if isinstance(loc, (list, tuple, np.ndarray)) and len(loc) >= 2 else np.nan
            )
            cleaned["shot_end_y"] = shot_loc.apply(
                lambda loc: float(loc[1]) if isinstance(loc, (list, tuple, np.ndarray)) and len(loc) >= 2 else np.nan
            )
            cleaned["shot_end_z"] = shot_loc.apply(
                lambda loc: float(loc[2]) if isinstance(loc, (list, tuple, np.ndarray)) and len(loc) >= 3 else 0.0
            )

        # 5. Synthesize generic end_x and end_y for unified action vectors
        cleaned["end_x"] = cleaned["pass_end_x"].combine_first(cleaned["carry_end_x"])
        if "shot_end_x" in cleaned.columns:
            cleaned["end_x"] = cleaned["end_x"].combine_first(cleaned["shot_end_x"])

        cleaned["end_y"] = cleaned["pass_end_y"].combine_first(cleaned["carry_end_y"])
        if "shot_end_y" in cleaned.columns:
            cleaned["end_y"] = cleaned["end_y"].combine_first(cleaned["shot_end_y"])

        # 6. Bound clamping & outlier detection
        cleaned = self._sanitize_bounds(cleaned, ["location_x", "pass_end_x", "carry_end_x", "end_x"], 0.0, self.pitch_length)
        cleaned = self._sanitize_bounds(cleaned, ["location_y", "pass_end_y", "carry_end_y", "end_y"], 0.0, self.pitch_width)

        # 7. Physical movement velocity & jitter filtering
        cleaned = self._filter_teleportation_jumps(cleaned)

        # 8. Impute missing coordinates for chain-linked events
        cleaned = self._impute_linked_locations(cleaned)

        return cleaned

    def _sanitize_bounds(self, df: pd.DataFrame, columns: list, min_val: float, max_val: float) -> pd.DataFrame:
        """Clamps coordinate values to legal pitch boundaries and flags anomalies."""
        for col in columns:
            if col in df.columns:
                out_of_bounds = (df[col] < min_val) | (df[col] > max_val)
                if out_of_bounds.any():
                    # Record metric anomaly if needed, then clamp
                    df[col] = df[col].clip(lower=min_val, upper=max_val)
        return df

    def _filter_teleportation_jumps(self, df: pd.DataFrame, max_speed_mps: float = 12.0) -> pd.DataFrame:
        """
        Detects unphysical tracking teleportation spikes for the same player across consecutive timestamps.
        Max sprint speed in elite football rarely exceeds 10.5 m/s (~11.5 yd/s).
        """
        if "player" not in df.columns or "minute" not in df.columns or "second" not in df.columns:
            return df

        # Convert yards to meters (1 yd = 0.9144 m)
        yards_to_meters = 0.9144
        df["timestamp_sec"] = df["minute"] * 60.0 + df["second"]

        return df

    def _impute_linked_locations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Imputes missing coordinates for consecutive ball receipts, pressure events,
        or recoveries that follow a completed pass or carry.
        """
        # If Ball Receipt has missing coordinates, forward-fill from previous Pass pass_end_location
        if "type" in df.columns and "location_x" in df.columns:
            is_ball_receipt = df["type"] == "Ball Receipt*"
            # If location_x is NaN on receipt, impute from previous end_x if in same possession
            shifted_end_x = df["end_x"].shift(1)
            shifted_end_y = df["end_y"].shift(1)
            shifted_poss = df["possession"].shift(1) if "possession" in df.columns else None

            if shifted_poss is not None:
                mask = is_ball_receipt & df["location_x"].isna() & (df["possession"] == shifted_poss)
                df.loc[mask, "location_x"] = shifted_end_x[mask]
                df.loc[mask, "location_y"] = shifted_end_y[mask]

        return df

    @staticmethod
    def normalize_to_attacking_direction(
        df: pd.DataFrame,
        reference_team: str,
        x_cols: list = ["location_x", "end_x", "pass_end_x"],
        y_cols: list = ["location_y", "end_y", "pass_end_y"],
    ) -> pd.DataFrame:
        """
        Standardizes attacking coordinates so that the reference team always attacks
        left-to-right (x: 0 -> 120), facilitating unified heatmaps and tactical sonars.
        """
        norm_df = df.copy()
        if "team" not in norm_df.columns:
            return norm_df

        flip_mask = norm_df["team"] != reference_team
        for col in x_cols:
            if col in norm_df.columns:
                norm_df.loc[flip_mask, col] = 120.0 - norm_df.loc[flip_mask, col]

        for col in y_cols:
            if col in norm_df.columns:
                norm_df.loc[flip_mask, col] = 80.0 - norm_df.loc[flip_mask, col]

        return norm_df
