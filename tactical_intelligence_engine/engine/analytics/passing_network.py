"""
Passing Network & Graph Centrality Analytics Engine.
Calculates graph topological density, betweenness centrality, tactical hub identification,
and passing efficiency ratios using NetworkX and Pandas.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import networkx as nx


class PassingNetworkEngine:
    """
    Constructs tactical passing graphs, computes network density,
    centrality indices, and pass efficiency profiles.
    """

    def __init__(self, min_pass_threshold: int = 3):
        self.min_pass_threshold = min_pass_threshold

    def compute_team_network(
        self,
        df: pd.DataFrame,
        team: str,
        first_n_minutes: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Calculates passing network structure, average player coordinates,
        graph density, degree and betweenness centrality.
        """
        # Filter completed passes by team
        passes = df[(df["team"] == team) & (df["type"] == "Pass")].copy()
        if first_n_minutes:
            passes = passes[passes["minute"] <= first_n_minutes]

        pass_complete = passes["pass_outcome"].isna() | (passes["pass_outcome"] == "Complete")
        completed_passes = passes[pass_complete].copy()

        if completed_passes.empty:
            return {
                "team": team,
                "network_density": 0.0,
                "avg_clustering": 0.0,
                "total_completed_passes": 0,
                "nodes": pd.DataFrame(),
                "edges": pd.DataFrame(),
                "graph": nx.DiGraph(),
                "efficiency": {}
            }

        # Calculate average spatial position for each player
        # In StatsBomb, player is in 'player', recipient is in 'pass_recipient'
        passer_locs = completed_passes[["player", "location_x", "location_y"]].rename(
            columns={"location_x": "x", "location_y": "y"}
        )
        receiver_locs = completed_passes[["pass_recipient", "end_x", "end_y"]].dropna().rename(
            columns={"pass_recipient": "player", "end_x": "x", "end_y": "y"}
        )
        combined_locs = pd.concat([passer_locs, receiver_locs], ignore_index=True)

        player_positions = combined_locs.groupby("player").agg(
            x_mean=("x", "mean"),
            y_mean=("y", "mean"),
            touches=("x", "count")
        ).reset_index()

        # Isolate Starting XI (top 11 players by touches) to prevent substitute node overlap
        if len(player_positions) > 11:
            top_11 = set(player_positions.sort_values(by="touches", ascending=False).head(11)["player"])
            player_positions = player_positions[player_positions["player"].isin(top_11)].copy()
        else:
            top_11 = set(player_positions["player"])

        # Build edge combinations (passer -> recipient)
        edge_data = completed_passes.groupby(["player", "pass_recipient"]).size().reset_index(name="pass_count")
        edge_data.rename(columns={"player": "source", "pass_recipient": "target"}, inplace=True)
        edge_data = edge_data.dropna()
        edge_data = edge_data[edge_data["source"].isin(top_11) & edge_data["target"].isin(top_11)].copy()

        # Build NetworkX Directed Graph
        G = nx.DiGraph()

        # Add nodes with attributes
        for _, row in player_positions.iterrows():
            G.add_node(
                row["player"],
                pos=(row["x_mean"], row["y_mean"]),
                touches=row["touches"]
            )

        # Add edges
        for _, row in edge_data.iterrows():
            G.add_edge(row["source"], row["target"], weight=row["pass_count"])

        # Graph Analytics
        density = nx.density(G)
        
        # Centrality metrics
        try:
            betweenness = nx.betweenness_centrality(G, weight="weight", normalized=True)
            degree_cent = nx.degree_centrality(G)
            clustering = nx.average_clustering(G.to_undirected(), weight="weight")
        except Exception:
            betweenness = {n: 0.0 for n in G.nodes()}
            degree_cent = {n: 0.0 for n in G.nodes()}
            clustering = 0.0

        # Attach centrality metrics to node DataFrame
        player_positions["betweenness"] = player_positions["player"].map(betweenness).fillna(0.0)
        player_positions["degree_centrality"] = player_positions["player"].map(degree_cent).fillna(0.0)

        # Passing Efficiency Metrics
        total_attempted = len(passes)
        total_completed = len(completed_passes)
        completion_pct = (total_completed / total_attempted * 100.0) if total_attempted > 0 else 0.0

        # Progressive pass share
        if "is_progressive_pass" in passes.columns:
            prog_passes = len(passes[passes["is_progressive_pass"] == True])
        else:
            prog_passes = 0
        prog_pct = (prog_passes / total_completed * 100.0) if total_completed > 0 else 0.0

        # Forward pass share (end_x > location_x)
        forward_passes = len(completed_passes[completed_passes["end_x"] > completed_passes["location_x"]])
        forward_pct = (forward_passes / total_completed * 100.0) if total_completed > 0 else 0.0

        # High pressure pass resistance
        if "under_pressure" in passes.columns:
            under_pressure = passes[passes["under_pressure"] == True]
            up_complete = len(under_pressure[under_pressure["pass_outcome"].isna()])
            up_completion_rate = (up_complete / len(under_pressure) * 100.0) if len(under_pressure) > 0 else 0.0
        else:
            up_completion_rate = 0.0

        efficiency = {
            "passes_attempted": total_attempted,
            "passes_completed": total_completed,
            "pass_completion_pct": round(completion_pct, 1),
            "progressive_passes": prog_passes,
            "progressive_pass_pct": round(prog_pct, 1),
            "forward_pass_pct": round(forward_pct, 1),
            "pressure_resistance_pct": round(up_completion_rate, 1),
        }

        # Filter edges for visual clarity (threshold)
        prominent_edges = edge_data[edge_data["pass_count"] >= self.min_pass_threshold].copy()

        return {
            "team": team,
            "network_density": round(density, 3),
            "avg_clustering": round(clustering, 3),
            "total_completed_passes": total_completed,
            "nodes": player_positions.sort_values(by="touches", ascending=False).reset_index(drop=True),
            "edges": prominent_edges.sort_values(by="pass_count", ascending=False).reset_index(drop=True),
            "all_edges": edge_data,
            "graph": G,
            "efficiency": efficiency,
            "tactical_hubs": player_positions.sort_values(by="betweenness", ascending=False).head(3)["player"].tolist()
        }
