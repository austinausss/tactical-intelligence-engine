"""
Shot Chart & Expected Goals (xG) Flow Visualization Engine.
Generates half-pitch coaching shot charts, xG bubble maps, outcome classifications,
and cumulative match xG flow timelines.
"""

from typing import Tuple, Optional, Any, Dict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

from config import STYLE_CONFIG
from engine.visualization.pitch import PitchDrawer


class ShotChartVisualizer:
    """
    Renders coaching staff shot maps and cumulative momentum timelines.
    """

    def __init__(self, theme: dict = STYLE_CONFIG):
        self.theme = theme
        self.pitch_drawer = PitchDrawer(theme=theme)

    def plot_shot_chart(
        self,
        df: pd.DataFrame,
        team: str,
        ax: Optional[plt.Axes] = None,
        efficiency_stats: Optional[Dict[str, Any]] = None,
        figsize: Tuple[int, int] = (10, 8),
        title: Optional[str] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plots a half-pitch shot chart with bubble sizes scaled to xG and colors
        indicating outcome (Goal, Saved, Off Target, Blocked).
        """
        shots = df[(df["team"] == team) & (df["type"] == "Shot")].copy()
        fig, ax = self.pitch_drawer.draw_pitch(ax=ax, figsize=figsize, half=True)

        if shots.empty:
            ax.text(90.0, 40.0, f"No shots recorded for {team}", color=self.theme["text_muted"], ha="center", va="center")
            return fig, ax

        # Ensure shots attack toward x=120
        # If any shots have x < 60, invert to half pitch
        if (shots["location_x"] < 60.0).any():
            shots["location_x"] = 120.0 - shots["location_x"]
            shots["location_y"] = 80.0 - shots["location_y"]

        outcome_color_map = {
            "Goal": self.theme["goal"],
            "Saved": self.theme["saved"],
            "Off T": self.theme["off_target"],
            "Blocked": self.theme["blocked"],
            "Post": self.theme["post"]
        }

        # Plot non-goals first, then goals on top
        non_goals = shots[shots["shot_outcome"] != "Goal"]
        goals = shots[shots["shot_outcome"] == "Goal"]

        for _, shot in non_goals.iterrows():
            sx = shot["location_x"]
            sy = shot["location_y"]
            xg = shot.get("xg", 0.1) or 0.1
            outcome = shot.get("shot_outcome", "Off T")
            color = outcome_color_map.get(outcome, self.theme["off_target"])

            bubble_size = 40.0 + (xg * 450.0)

            ax.scatter(
                sx, sy,
                s=bubble_size,
                facecolor=color,
                edgecolor="#ffffff",
                lw=0.8,
                alpha=0.82,
                zorder=4
            )

        # Plot goals with prominence
        for _, shot in goals.iterrows():
            sx = shot["location_x"]
            sy = shot["location_y"]
            xg = shot.get("xg", 0.25) or 0.25
            bubble_size = 80.0 + (xg * 550.0)
            player = shot.get("player", "Goal")
            minute = shot.get("minute", 0)

            # Glowing star/marker
            ax.scatter(
                sx, sy,
                s=bubble_size,
                facecolor=self.theme["goal"],
                edgecolor="#ffffff",
                lw=2.2,
                alpha=0.98,
                zorder=6
            )
            # Annotate scorer and minute
            parts = player.split()
            surname = parts[-1] if len(parts) > 1 else player
            ax.annotate(
                f"GOAL: {surname} ({minute}')\nxG: {xg:.2f}",
                xy=(sx, sy),
                xytext=(sx - 3.5, sy - 4.5),
                fontsize=8.0,
                fontweight="bold",
                color="#ffffff",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#064e3b", edgecolor="#10b981", alpha=0.9),
                arrowprops=dict(arrowstyle="->", color="#10b981", lw=1.0),
                zorder=7
            )

        # Header Title (Above pitch with clean pad)
        chart_title = title or f"{team.upper()} • SHOT QUALITY & xG"
        ax.set_title(chart_title, color=self.theme["text_color"], fontsize=11.5, fontweight="bold", pad=12)

        # Coaching Staff KPI Box
        if efficiency_stats:
            kpi_text = (
                f"Goals: {efficiency_stats.get('goals', 0)}  |  Total Shots: {efficiency_stats.get('total_shots', 0)}\n"
                f"Expected Goals (xG): {efficiency_stats.get('total_xg', 0.0):.2f}\n"
                f"Goals / xG Ratio: {efficiency_stats.get('xg_efficiency_ratio', 0.0):.2f}x\n"
                f"Shot Conversion: {efficiency_stats.get('shot_conversion_rate', 0.0)}%\n"
                f"On-Target Acc: {efficiency_stats.get('on_target_percentage', 0.0)}%\n"
                f"Rating: {efficiency_stats.get('clinical_rating', 'N/A')[:22]}"
            )
            ax.text(
                118.0, 76.5, kpi_text,
                color=self.theme["text_color"], fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#1e293b", edgecolor=self.theme["pitch_lines"], alpha=0.9),
                va="bottom", ha="right", zorder=8
            )

        # Legend
        legend_elements = [
            patches.Patch(facecolor=self.theme["goal"], edgecolor="#ffffff", label="Goal"),
            patches.Patch(facecolor=self.theme["saved"], edgecolor="#ffffff", label="Saved"),
            patches.Patch(facecolor=self.theme["off_target"], edgecolor="#ffffff", label="Off Target"),
            patches.Patch(facecolor=self.theme["blocked"], edgecolor="#ffffff", label="Blocked"),
        ]
        leg = ax.legend(
            handles=legend_elements, loc="lower left",
            facecolor="#1e293b", edgecolor=self.theme["pitch_lines"],
            fontsize=7.5, labelcolor=self.theme["text_color"]
        )
        leg.set_zorder(8)

        return fig, ax

    def plot_cumulative_xg_timeline(
        self,
        df: pd.DataFrame,
        team_a: str,
        team_b: str,
        ax: Optional[plt.Axes] = None,
        figsize: Tuple[int, int] = (12, 4.5),
        save_path: Optional[str] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plots the cumulative Expected Goals (xG) accumulation timeline across match minutes.
        """
        shots = df[df["type"] == "Shot"].sort_values(by=["minute", "second"]).copy()

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, facecolor=self.theme["bg_color"])
        else:
            fig = ax.figure
        ax.set_facecolor(self.theme["pitch_color"])

        max_min = max(90, int(shots["minute"].max() or 90) + 5)
        timeline = np.arange(0, max_min + 1)

        # Build step-wise cumulative xG series
        def build_cum_xg(team_name: str):
            t_shots = shots[shots["team"] == team_name]
            xg_series = [0.0] * len(timeline)
            cum = 0.0
            shot_idx = 0
            t_shot_list = t_shots.to_dict("records")
            
            for m in timeline:
                while shot_idx < len(t_shot_list) and t_shot_list[shot_idx]["minute"] <= m:
                    cum += float(t_shot_list[shot_idx].get("xg", 0.1) or 0.1)
                    shot_idx += 1
                xg_series[m] = cum
            return xg_series, t_shots

        xg_a, shots_a = build_cum_xg(team_a)
        xg_b, shots_b = build_cum_xg(team_b)

        col_a = self.theme["argentina_primary"] if "Argentina" in team_a else self.theme["spain_primary"]
        col_b = self.theme["spain_primary"] if "Spain" in team_b else self.theme["argentina_primary"]

        ax.step(timeline, xg_a, color=col_a, lw=2.5, label=f"{team_a} (xG: {xg_a[-1]:.2f})", zorder=3)
        ax.step(timeline, xg_b, color=col_b, lw=2.5, label=f"{team_b} (xG: {xg_b[-1]:.2f})", zorder=3)

        # Plot Goal markers
        for _, shot in shots[shots["shot_outcome"] == "Goal"].iterrows():
            m = shot["minute"]
            p = shot["player"]
            t = shot["team"]
            xg_val = xg_a[m] if t == team_a else xg_b[m]
            marker_col = col_a if t == team_a else col_b

            ax.scatter(m, xg_val, color=self.theme["goal"], edgecolor="#ffffff", s=110, lw=1.8, zorder=5)
            surname = p.split()[-1] if len(p.split()) > 1 else p
            ax.annotate(
                f"GOAL: {surname} ({m}')",
                xy=(m, xg_val),
                xytext=(m - 8 if m > 90 else m, xg_val + 0.38),
                fontsize=8.5, fontweight="bold", color="#ffffff",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#1e293b", edgecolor=marker_col, alpha=0.9),
                ha="right" if m > 90 else "center",
                arrowprops=dict(arrowstyle="->", color="#ffffff", lw=0.9),
                zorder=6
            )

        # Vertical line for Half-time, 90 min, Extra-time
        ax.axvline(45, color=self.theme["pitch_lines"], linestyle="--", alpha=0.6, zorder=2)
        ax.text(45.5, 0.1, "HT", color=self.theme["text_muted"], fontsize=8)

        if max_min > 90:
            ax.axvline(90, color=self.theme["pitch_lines"], linestyle="--", alpha=0.6, zorder=2)
            ax.text(90.5, 0.1, "FT (AET)", color=self.theme["text_muted"], fontsize=8)

        ax.set_xlim(0, max_min)
        ax.set_ylim(0, max(2.6, max(max(xg_a), max(xg_b)) + 0.9))
        ax.set_xlabel("Match Time (Minutes)", color=self.theme["text_muted"], fontsize=10)
        ax.set_ylabel("Cumulative Expected Goals (xG)", color=self.theme["text_muted"], fontsize=10)
        ax.set_title("MATCH MOMENTUM & EXPECTED GOALS (xG) TRAJECTORY", color=self.theme["text_color"], fontsize=12, fontweight="bold", pad=12)

        ax.tick_params(colors=self.theme["text_muted"])
        for spine in ax.spines.values():
            spine.set_color(self.theme["pitch_lines"])

        leg = ax.legend(loc="upper left", facecolor="#1e293b", edgecolor=self.theme["pitch_lines"], labelcolor=self.theme["text_color"])
        leg.set_zorder(7)
        if save_path or ax is None:
            plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, facecolor=self.theme["bg_color"], bbox_inches="tight")

        return fig, ax
