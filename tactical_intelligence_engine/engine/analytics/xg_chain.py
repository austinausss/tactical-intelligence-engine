"""
Expected Goals Chain (xGChain) & Buildup (xGBuildup) Analytics Engine.
Evaluates total possession sequence contribution, isolating deep-lying playmakers,
progressive connectors, and buildup architects.
"""

from typing import Dict, List, Optional, Tuple, Set
import numpy as np
import pandas as pd


class XGChainEngine:
    """
    Computes xGChain and xGBuildup across possession sequences.
    
    xGChain: Sum of xG of all possession sequences in which a player was involved.
    xGBuildup: Sum of xG of possession sequences in which a player was involved,
               excluding the shot taker and the direct shot assist creator.
    """

    def __init__(self):
        pass

    def compute_chain_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates individual player xGChain, xGBuildup, and sequence involvement rates.
        """
        events = df.copy()

        # Check required columns
        if "possession" not in events.columns or "player" not in events.columns:
            return pd.DataFrame()

        # Grouping keys
        group_keys = ["match_id", "period", "possession"]
        # Ensure match_id and period exist
        for col in ["match_id", "period"]:
            if col not in events.columns:
                events[col] = 1

        # Identify shots and their xG per possession chain
        shots = events[events["type"] == "Shot"].copy()
        if "xg" not in shots.columns and "shot_statsbomb_xg" in shots.columns:
            shots["xg"] = shots["shot_statsbomb_xg"].fillna(0.1)
        elif "xg" not in shots.columns:
            shots["xg"] = 0.1

        # Map each possession chain to total shot xG and key actors
        chain_shot_map: Dict[Tuple, Dict[str, Any]] = {}

        for _, shot in shots.iterrows():
            chain_id = (shot["match_id"], shot["period"], shot["possession"])
            shot_xg = float(shot["xg"]) if not pd.isna(shot["xg"]) else 0.05
            shooter = shot["player"]

            if chain_id not in chain_shot_map:
                chain_shot_map[chain_id] = {
                    "total_xg": 0.0,
                    "shooters": set(),
                    "assisters": set(),
                }
            chain_shot_map[chain_id]["total_xg"] += shot_xg
            if pd.notna(shooter):
                chain_shot_map[chain_id]["shooters"].add(shooter)

        # Identify shot assist / key pass creators
        pass_assists = events[events["type"] == "Pass"].copy()
        if "pass_assisted_shot_id" in pass_assists.columns or "pass_shot_assist" in pass_assists.columns:
            assist_mask = (
                pass_assists["pass_assisted_shot_id"].notna() |
                (pass_assists.get("pass_shot_assist") == True)
            )
            for _, p in pass_assists[assist_mask].iterrows():
                chain_id = (p["match_id"], p["period"], p["possession"])
                if chain_id in chain_shot_map and pd.notna(p["player"]):
                    chain_shot_map[chain_id]["assisters"].add(p["player"])
        else:
            # Fallback heuristic: the pass directly preceding the shot in the same possession
            for chain_id in chain_shot_map:
                chain_events = events[
                    (events["match_id"] == chain_id[0]) &
                    (events["period"] == chain_id[1]) &
                    (events["possession"] == chain_id[2])
                ]
                passes = chain_events[chain_events["type"] == "Pass"]
                if not passes.empty:
                    last_passer = passes.iloc[-1]["player"]
                    if pd.notna(last_passer):
                        chain_shot_map[chain_id]["assisters"].add(last_passer)

        # Iterate over all player involvements in shot-producing chains
        player_stats: Dict[str, Dict[str, Any]] = {}

        # Pre-populate all players
        for player in events["player"].dropna().unique():
            p_events = events[events["player"] == player]
            p_team = p_events["team"].iloc[0] if "team" in p_events.columns else "Unknown"
            player_stats[player] = {
                "player": player,
                "team": p_team,
                "xg_chain": 0.0,
                "xg_buildup": 0.0,
                "shot_involvements": 0,
                "buildup_involvements": 0,
                "total_actions": len(p_events),
            }

        # Attribute xG to chain participants
        for chain_id, chain_info in chain_shot_map.items():
            chain_events = events[
                (events["match_id"] == chain_id[0]) &
                (events["period"] == chain_id[1]) &
                (events["possession"] == chain_id[2])
            ]
            chain_players = set(chain_events["player"].dropna().unique())
            c_xg = chain_info["total_xg"]
            shooters = chain_info["shooters"]
            assisters = chain_info["assisters"]

            for player in chain_players:
                if player in player_stats:
                    player_stats[player]["xg_chain"] += c_xg
                    player_stats[player]["shot_involvements"] += 1

                    # Buildup check: player is not the shooter and not the assister
                    if player not in shooters and player not in assisters:
                        player_stats[player]["xg_buildup"] += c_xg
                        player_stats[player]["buildup_involvements"] += 1

        # Format into DataFrame
        result_df = pd.DataFrame(list(player_stats.values()))
        result_df["xg_chain"] = result_df["xg_chain"].round(3)
        result_df["xg_buildup"] = result_df["xg_buildup"].round(3)

        # Buildup ratio (% of xGChain coming purely from non-terminal buildup)
        result_df["buildup_share_pct"] = np.where(
            result_df["xg_chain"] > 0,
            (result_df["xg_buildup"] / result_df["xg_chain"] * 100.0).round(1),
            0.0
        )

        return result_df.sort_values(by="xg_chain", ascending=False).reset_index(drop=True)
