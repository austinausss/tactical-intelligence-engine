"""
Expected Goals (xG) & Shot Efficiency Intelligence Engine.
Computes calibrated xG values, shot efficiency ratios, clinical conversion indices,
and zone-based finishing performance.
"""

from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from config import PITCH_LENGTH, PITCH_WIDTH, GOAL_CENTER


class XGEngine:
    """
    Evaluates shot quality and finishing efficiency across teams and individual players.
    """

    def __init__(self, pitch_length: float = PITCH_LENGTH, pitch_width: float = PITCH_WIDTH):
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width

    def calibrate_or_fill_xg(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensures every shot record has a valid Expected Goals (xG) value.
        If StatsBomb xG is absent, applies a calibrated geometric-spatial regression model.
        """
        df = df.copy()
        if "type" not in df.columns:
            return df

        is_shot = df["type"] == "Shot"
        if not is_shot.any():
            return df

        # Ensure spatial columns
        x = df["location_x"]
        y = df["location_y"]

        # Calculate geometric distance to target goal (either x=120 or x=0)
        dx_right = np.abs(self.pitch_length - x)
        dx_left = np.abs(x)
        dx = np.minimum(dx_right, dx_left)
        dy = 40.0 - y
        dist = np.sqrt(dx**2 + dy**2)

        # Visible goal angle in radians
        angle1 = np.arctan2(44.0 - y, np.maximum(dx, 0.1))
        angle2 = np.arctan2(36.0 - y, np.maximum(dx, 0.1))
        shot_angle = np.abs(angle1 - angle2)

        # Calibrated logistic model: P(Goal) = 1 / (1 + exp(-(b0 + b1*dist + b2*angle)))
        dist_scaled = dist * 0.9144  # to meters
        logit = -1.25 - (0.165 * dist_scaled) + (1.95 * shot_angle)
        calibrated_xg = 1.0 / (1.0 + np.exp(-logit))

        # Check existing xG
        if "shot_statsbomb_xg" in df.columns:
            df["xg"] = df["shot_statsbomb_xg"].fillna(calibrated_xg)
        else:
            df["xg"] = np.where(is_shot, calibrated_xg, 0.0)

        df["xg"] = df["xg"].clip(lower=0.01, upper=0.98)
        return df

    def compute_team_shot_efficiency(self, df: pd.DataFrame, team: str) -> Dict[str, Any]:
        """
        Computes comprehensive shot efficiency ratios and finishing quality metrics for a team.
        """
        team_shots = df[(df["team"] == team) & (df["type"] == "Shot")].copy()
        
        if team_shots.empty:
            return {
                "team": team,
                "total_shots": 0,
                "shots_on_target": 0,
                "goals": 0,
                "total_xg": 0.0,
                "xg_per_shot": 0.0,
                "shot_conversion_rate": 0.0,
                "on_target_percentage": 0.0,
                "on_target_conversion": 0.0,
                "xg_efficiency_ratio": 0.0,
                "box_shot_ratio": 0.0,
                "clinical_rating": "N/A"
            }

        total_shots = len(team_shots)
        goals = len(team_shots[team_shots["shot_outcome"] == "Goal"])
        on_target = len(team_shots[team_shots["shot_outcome"].isin(["Goal", "Saved"])])
        total_xg = float(team_shots["xg"].sum()) if "xg" in team_shots.columns else 0.0

        xg_per_shot = total_xg / total_shots if total_shots > 0 else 0.0
        conversion_rate = (goals / total_shots * 100.0) if total_shots > 0 else 0.0
        on_target_pct = (on_target / total_shots * 100.0) if total_shots > 0 else 0.0
        on_target_conv = (goals / on_target * 100.0) if on_target > 0 else 0.0

        # xG Efficiency Ratio: Goals / xG (1.0 = expected, >1.0 = clinical, <1.0 = wasteful)
        xg_efficiency = (goals / total_xg) if total_xg > 0 else 0.0

        # Shots from inside 18-yard box
        inside_box = team_shots[
            (team_shots["location_x"] >= 102.0) &
            (team_shots["location_y"] >= 18.0) &
            (team_shots["location_y"] <= 62.0)
        ]
        box_ratio = (len(inside_box) / total_shots * 100.0) if total_shots > 0 else 0.0

        # Clinical rating classification
        if xg_efficiency >= 1.35:
            rating = "Elite Clinical Finishing (High Overperformance)"
        elif xg_efficiency >= 1.05:
            rating = "High Efficiency (Positive Overperformance)"
        elif xg_efficiency >= 0.85:
            rating = "Expected Finishing Efficiency (Par)"
        else:
            rating = "Sub-Optimal Finishing (Underperforming xG)"

        return {
            "team": team,
            "total_shots": total_shots,
            "shots_on_target": on_target,
            "goals": goals,
            "total_xg": round(total_xg, 2),
            "xg_per_shot": round(xg_per_shot, 3),
            "shot_conversion_rate": round(conversion_rate, 1),
            "on_target_percentage": round(on_target_pct, 1),
            "on_target_conversion": round(on_target_conv, 1),
            "xg_efficiency_ratio": round(xg_efficiency, 2),
            "box_shot_ratio": round(box_ratio, 1),
            "clinical_rating": rating
        }

    def compute_player_shot_rankings(self, df: pd.DataFrame, team: Optional[str] = None) -> pd.DataFrame:
        """Ranks players by shot volume, xG generated, goals, and conversion efficiency."""
        shots = df[df["type"] == "Shot"].copy()
        if team:
            shots = shots[shots["team"] == team]

        if shots.empty:
            return pd.DataFrame()

        summary = []
        for player, group in shots.groupby("player"):
            p_team = group["team"].iloc[0]
            tot_shots = len(group)
            goals = len(group[group["shot_outcome"] == "Goal"])
            tot_xg = float(group["xg"].sum()) if "xg" in group.columns else 0.0
            on_tgt = len(group[group["shot_outcome"].isin(["Goal", "Saved"])])
            
            summary.append({
                "player": player,
                "team": p_team,
                "shots": tot_shots,
                "on_target": on_tgt,
                "goals": goals,
                "total_xg": round(tot_xg, 2),
                "xg_per_shot": round(tot_xg / tot_shots, 3) if tot_shots > 0 else 0.0,
                "conversion_pct": round(goals / tot_shots * 100.0, 1) if tot_shots > 0 else 0.0,
                "goals_minus_xg": round(goals - tot_xg, 2)
            })

        ranking_df = pd.DataFrame(summary).sort_values(by=["goals", "total_xg"], ascending=False).reset_index(drop=True)
        return ranking_df
