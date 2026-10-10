"""
Player Pitch Heatmap & Spatial Density Engine.
Generates 2D Gaussian Kernel Density Estimation (KDE) and action zone heatmaps
for coaching staff visual scouting reports.
"""

from typing import Tuple, Optional, Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import seaborn as sns

from config import STYLE_CONFIG
from engine.visualization.pitch import PitchDrawer


class HeatmapVisualizer:
    """
    Creates high-contrast spatial density heatmaps and activity distribution plots.
    """

    def __init__(self, theme: dict = STYLE_CONFIG):
        self.theme = theme
        self.pitch_drawer = PitchDrawer(theme=theme)

    def plot_player_heatmap(
        self,
        df: pd.DataFrame,
        player_name: str,
        ax: Optional[plt.Axes] = None,
        cmap: str = "magma",
        show_stats: bool = True
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Renders a 2D Gaussian KDE spatial density heatmap for a specific player.
        """
        player_events = df[
            (df["player"] == player_name) &
            (df["location_x"].notna()) &
            (df["location_y"].notna())
        ].copy()

        fig, ax = self.pitch_drawer.draw_pitch(ax=ax, figsize=(11, 7.5))

        if len(player_events) < 5:
            ax.text(
                60.0, 40.0, f"Insufficient spatial actions for {player_name}",
                color=self.theme["text_muted"], ha="center", va="center", fontsize=12
            )
            return fig, ax

        x = player_events["location_x"].values
        y = player_events["location_y"].values

        # Perform 2D KDE using seaborn for smooth aesthetic contouring
        try:
            sns.kdeplot(
                x=x,
                y=y,
                ax=ax,
                cmap=cmap,
                fill=True,
                thresh=0.08,
                levels=45,
                alpha=0.65,
                zorder=3
            )
        except Exception:
            # Fallback scatter if KDE has low variance
            ax.scatter(x, y, color=self.theme["accent_primary"], alpha=0.5, s=25, zorder=3)

        # Plot individual touch points with subtle glow
        ax.scatter(x, y, color="#ffffff", alpha=0.25, s=12, edgecolors="none", zorder=4)

        # Team & Position info
        team = player_events["team"].iloc[0] if "team" in player_events.columns else ""
        pos = player_events["position"].iloc[0] if "position" in player_events.columns else ""
        tot_touches = len(player_events)
        is_spain = "spain" in team.lower()
        if is_spain:
            f3_touches = len(player_events[player_events["location_x"] <= 40.0])
        else:
            f3_touches = len(player_events[player_events["location_x"] >= 80.0])
        f3_share = (f3_touches / tot_touches * 100.0) if tot_touches > 0 else 0.0

        # Header Title (Clean above pitch, no overlap)
        title_text = f"{player_name.upper()} ({team.upper()}) • TERRITORIAL DENSITY"
        ax.set_title(title_text, color=self.theme["text_color"], fontsize=12, fontweight="bold", pad=12)

        if show_stats:
            stats_box = (
                f"Total Actions: {tot_touches}\n"
                f"Final Third Actions: {f3_touches} ({f3_share:.1f}%)\n"
                f"Avg Position: ({x.mean():.1f}, {y.mean():.1f})"
            )
            ax.text(
                116.0, 75.0, stats_box,
                color=self.theme["text_color"], fontsize=8.0,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#1e293b", edgecolor=self.theme["pitch_lines"], alpha=0.88),
                va="bottom", ha="right", zorder=5
            )

        return fig, ax

    def plot_head_to_head_heatmaps(
        self,
        df: pd.DataFrame,
        player_a: str,
        player_b: str,
        figsize: Tuple[int, int] = (16, 7.5),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Creates a dual-panel scouting graphic comparing heatmaps of two talismanic players
        (e.g., Lionel Messi vs Lamine Yamal).
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize, facecolor=self.theme["bg_color"])
        
        self.plot_player_heatmap(df, player_a, ax=axes[0], cmap="Blues_r")
        self.plot_player_heatmap(df, player_b, ax=axes[1], cmap="Reds_r")

        plt.subplots_adjust(wspace=0.05, top=0.88, bottom=0.05, left=0.02, right=0.98)

        fig.suptitle(
            f"TACTICAL MATCHUP: {player_a.upper()} vs {player_b.upper()}",
            color=self.theme["text_color"], fontsize=16, fontweight="bold", y=0.96
        )

        if save_path:
            fig.savefig(save_path, dpi=300, facecolor=self.theme["bg_color"], bbox_inches="tight")
            print(f"[HeatmapVisualizer] Saved head-to-head scouting report to {save_path}")

        return fig
