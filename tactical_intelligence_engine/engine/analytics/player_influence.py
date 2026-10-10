"""
Player Influence, Field Tilt & Territorial Dominance Engine.
Evaluates territorial control (Field Tilt), progressive ball progression,
high-press defensive workrate, and holistic player influence scores.
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from config import FINAL_THIRD_X, PITCH_LENGTH


class PlayerInfluenceEngine:
    """
    Computes Field Tilt, Progressive Action generation, Defensive disruption,
    and Composite Player Influence ratings.
    """

    def __init__(self, final_third_x: float = FINAL_THIRD_X):
        self.final_third_x = final_third_x

    def compute_match_field_tilt(self, df: pd.DataFrame, team_a: str, team_b: str) -> Dict[str, Any]:
        """
        Calculates Field Tilt (% share of final-third touches/passes) between two opposing teams.
        Field Tilt measures territorial dominance and pitch compression.
        """
        # Touches/passes in opponent final third (Direction-aware)
        is_a_spain = team_a == "Spain"
        is_b_spain = team_b == "Spain"
        mask_a = (df["location_x"] <= 40.0) if is_a_spain else (df["location_x"] >= self.final_third_x)
        mask_b = (df["location_x"] <= 40.0) if is_b_spain else (df["location_x"] >= self.final_third_x)

        f3_actions_a = len(df[(df["team"] == team_a) & mask_a])
        f3_actions_b = len(df[(df["team"] == team_b) & mask_b])

        total_f3 = f3_actions_a + f3_actions_b
        tilt_a = (f3_actions_a / total_f3 * 100.0) if total_f3 > 0 else 50.0
        tilt_b = (f3_actions_b / total_f3 * 100.0) if total_f3 > 0 else 50.0

        # General possession % comparison
        poss_a = len(df[df["possession_team"] == team_a]) if "possession_team" in df.columns else len(df[df["team"] == team_a])
        poss_b = len(df[df["possession_team"] == team_b]) if "possession_team" in df.columns else len(df[df["team"] == team_b])
        total_poss = poss_a + poss_b
        poss_pct_a = (poss_a / total_poss * 100.0) if total_poss > 0 else 50.0
        poss_pct_b = (poss_b / total_poss * 100.0) if total_poss > 0 else 50.0

        return {
            "team_a": team_a,
            "team_b": team_b,
            "field_tilt_a": round(tilt_a, 1),
            "field_tilt_b": round(tilt_b, 1),
            "possession_a": round(poss_pct_a, 1),
            "possession_b": round(poss_pct_b, 1),
            "f3_actions_a": f3_actions_a,
            "f3_actions_b": f3_actions_b,
            "territorial_verdict": f"{team_a if tilt_a > tilt_b else team_b} dominated attacking territory ({max(tilt_a, tilt_b):.1f}% Field Tilt)"
        }

    def compute_player_influence_profiles(
        self,
        df: pd.DataFrame,
        xg_chain_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Produces holistic player impact rankings blending progressive actions,
        possession orchestration (xGChain), and defensive pressure.
        """
        records = []
        for player, group in df.groupby("player"):
            team = group["team"].iloc[0]
            pos = group["position"].iloc[0] if "position" in group.columns else "Player"
            
            # Action volumes
            total_actions = len(group)
            passes = group[group["type"] == "Pass"]
            carries = group[group["type"] == "Carry"]
            
            # Progressive metrics
            prog_passes = len(passes[passes["is_progressive_pass"] == True]) if "is_progressive_pass" in passes.columns else 0
            prog_carries = len(carries[carries["is_progressive_carry"] == True]) if "is_progressive_carry" in carries.columns else 0
            box_entries = len(group[group["is_box_entry"] == True]) if "is_box_entry" in group.columns else 0
            
            # Final third touches (Direction-aware)
            if team == "Spain":
                f3_touches = len(group[group["location_x"] <= 40.0]) if "location_x" in group.columns else 0
            else:
                f3_touches = len(group[group["location_x"] >= self.final_third_x]) if "location_x" in group.columns else 0

            # Defensive disruptions
            def_actions = len(group[group["is_defensive_action"] == True]) if "is_defensive_action" in group.columns else 0
            high_press = len(group[group["is_high_press"] == True]) if "is_high_press" in group.columns else 0

            records.append({
                "player": player,
                "team": team,
                "position": pos,
                "total_actions": total_actions,
                "progressive_passes": prog_passes,
                "progressive_carries": prog_carries,
                "total_progressive_actions": prog_passes + prog_carries,
                "box_entries": box_entries,
                "final_third_touches": f3_touches,
                "defensive_actions": def_actions,
                "high_press_actions": high_press,
            })

        profile_df = pd.DataFrame(records)

        # Merge xGChain if available
        if xg_chain_df is not None and not xg_chain_df.empty:
            profile_df = profile_df.merge(
                xg_chain_df[["player", "xg_chain", "xg_buildup"]],
                on="player",
                how="left"
            ).fillna({"xg_chain": 0.0, "xg_buildup": 0.0})
        else:
            profile_df["xg_chain"] = 0.0
            profile_df["xg_buildup"] = 0.0

        # Calculate Composite Influence Index (0 - 100 normalized)
        # Weights: xGChain (35%), Progressive Actions (30%), Box Entries & F3 (20%), Defensive (15%)
        max_chain = profile_df["xg_chain"].max() or 1.0
        max_prog = profile_df["total_progressive_actions"].max() or 1.0
        max_f3 = profile_df["final_third_touches"].max() or 1.0
        max_def = profile_df["defensive_actions"].max() or 1.0

        score = (
            (profile_df["xg_chain"] / max_chain * 35.0) +
            (profile_df["total_progressive_actions"] / max_prog * 30.0) +
            (profile_df["final_third_touches"] / max_f3 * 20.0) +
            (profile_df["defensive_actions"] / max_def * 15.0)
        )
        profile_df["influence_score"] = score.round(1)

        return profile_df.sort_values(by="influence_score", ascending=False).reset_index(drop=True)
