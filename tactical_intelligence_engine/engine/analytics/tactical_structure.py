"""
Team Tactical Structure, Spatial Geometry & Defensive Block Engine.
Evaluates defensive line height, team compactness, spatial width, and convex hull territory.
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from config import PITCH_LENGTH, PITCH_WIDTH


class TacticalStructureEngine:
    """
    Measures physical team geometry: defensive line height, tactical width,
    vertical compression, and defensive block compactness.
    """

    def __init__(self, pitch_length: float = PITCH_LENGTH, pitch_width: float = PITCH_WIDTH):
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width

    def evaluate_team_shape(self, df: pd.DataFrame, team: str) -> Dict[str, Any]:
        """
        Calculates spatial metrics for a team's outfield players.
        """
        team_events = df[(df["team"] == team) & (df["location_x"].notna()) & (df["location_y"].notna())].copy()
        
        # Filter out Goalkeeper to assess outfield block geometry
        if "position" in team_events.columns:
            outfield = team_events[team_events["position"] != "Goalkeeper"]
        else:
            outfield = team_events

        if len(outfield) < 15:
            return {"team": team, "error": "Insufficient spatial event points."}

        # Spatial Centroid (Center of gravity)
        center_x = float(outfield["location_x"].mean())
        center_y = float(outfield["location_y"].mean())

        # Tactical Width (Spread along y-axis)
        width_std = float(outfield["location_y"].std())
        effective_width = float(outfield["location_y"].quantile(0.90) - outfield["location_y"].quantile(0.10))

        # Vertical Length / Depth (Spread along x-axis)
        length_std = float(outfield["location_x"].std())
        effective_length = float(outfield["location_x"].quantile(0.90) - outfield["location_x"].quantile(0.10))

        # Defensive Line Height: average x location of defensive actions in the backline
        defenders = outfield[outfield.get("position", "").str.contains("Back|Center Back", case=False, na=False)]
        if not defenders.empty:
            def_line_height = float(defenders["location_x"].mean())
        else:
            def_line_height = float(outfield["location_x"].quantile(0.25))

        # Convex Hull Area of outfield player centroids
        player_coords = outfield.groupby("player")[["location_x", "location_y"]].mean().values
        if len(player_coords) >= 4:
            try:
                hull = ConvexHull(player_coords)
                hull_area = float(hull.volume)  # In 2D, volume represents area (sq yards)
            except Exception:
                hull_area = effective_width * effective_length * 0.5
        else:
            hull_area = effective_width * effective_length * 0.5

        # Compactness ratio: Area relative to full pitch area (120 * 80 = 9600)
        pitch_area = self.pitch_length * self.pitch_width
        compactness_pct = (hull_area / pitch_area * 100.0)

        # Tactical Structure Classification
        if def_line_height >= 52.0:
            block_type = "Aggressive High-Line Pressing Block"
        elif def_line_height >= 40.0:
            block_type = "Disciplined Mid-Block Structure"
        else:
            block_type = "Deep Low-Block Defensive Wall"

        return {
            "team": team,
            "centroid_x": round(center_x, 1),
            "centroid_y": round(center_y, 1),
            "defensive_line_height": round(def_line_height, 1),
            "effective_width_yards": round(effective_width, 1),
            "effective_length_yards": round(effective_length, 1),
            "hull_area_sq_yards": round(hull_area, 1),
            "pitch_occupancy_pct": round(compactness_pct, 1),
            "tactical_archetype": block_type
        }
