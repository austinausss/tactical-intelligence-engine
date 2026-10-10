"""
Passing Network, Directional Sonar & Flow Vector Visualization Engine.
Visualizes tactical graph structures, passing combinations, directional flow vectors,
and progressive penetration channels for coaching staff.
"""

from typing import Tuple, Optional, Any, Dict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

from config import STYLE_CONFIG
from engine.visualization.pitch import PitchDrawer


class PassMapVisualizer:
    """
    Renders passing networks, directional pass flow sonars, and progressive pass maps.
    """

    def __init__(self, theme: dict = STYLE_CONFIG):
        self.theme = theme
        self.pitch_drawer = PitchDrawer(theme=theme)

    def plot_passing_network(
        self,
        network_data: Dict[str, Any],
        ax: Optional[plt.Axes] = None,
        node_color: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 8),
        title: Optional[str] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Renders a tactical passing network showing average player coordinates,
        inter-player pass combination arteries, and betweenness centrality.
        """
        fig, ax = self.pitch_drawer.draw_pitch(ax=ax, figsize=figsize)

        nodes = network_data.get("nodes", pd.DataFrame())
        edges = network_data.get("edges", pd.DataFrame())
        team = network_data.get("team", "Team")
        density = network_data.get("network_density", 0.0)
        eff = network_data.get("efficiency", {})

        if nodes.empty:
            ax.text(60.0, 40.0, "No passing data available", color=self.theme["text_muted"], ha="center", va="center")
            return fig, ax

        # Create quick coordinate lookup
        pos_map = {row["player"]: (row["x_mean"], row["y_mean"]) for _, row in nodes.iterrows()}

        # 1. Draw Passing Edges (Combinations)
        if not edges.empty:
            max_passes = edges["pass_count"].max() or 1
            for _, edge in edges.iterrows():
                u, v = edge["source"], edge["target"]
                if u in pos_map and v in pos_map:
                    x1, y1 = pos_map[u]
                    x2, y2 = pos_map[v]
                    count = edge["pass_count"]

                    # Scale width and alpha
                    lw = 1.0 + (count / max_passes) * 4.5
                    alpha = 0.25 + (count / max_passes) * 0.65

                    ax.plot([x1, x2], [y1, y2], color="#94a3b8", lw=lw, alpha=alpha, zorder=3)

        # 2. Draw Player Nodes
        team_color = node_color or (
            self.theme["argentina_primary"] if "Argentina" in team else self.theme["spain_primary"]
        )

        max_touches = nodes["touches"].max() or 1
        for _, node in nodes.iterrows():
            px, py = node["x_mean"], node["y_mean"]
            p_name = node["player"]
            touches = node["touches"]
            betweenness = node.get("betweenness", 0.0)

            # Node size scaled to touch volume
            radius = 2.0 + (touches / max_touches) * 2.8

            # Circle patch with glowing edge
            circle = patches.Circle(
                (px, py), radius,
                facecolor=team_color,
                edgecolor="#ffffff",
                lw=1.5,
                alpha=0.92,
                zorder=4
            )
            ax.add_patch(circle)

            # Professional Football Display Names (Official Shirt Names)
            KNOWN_DISPLAY_NAMES = {
                # Argentina Starting XI & Subs
                "Emiliano Martinez": "E. Martínez",
                "Emiliano Martínez": "E. Martínez",
                "Damián Emiliano Martínez": "E. Martínez",
                "Nahuel Molina": "Molina",
                "Nahuel Molina Lucero": "Molina",
                "Cristian Romero": "C. Romero",
                "Cristian Gabriel Romero": "C. Romero",
                "Lisandro Martinez": "L. Martínez",
                "Lisandro Martínez": "L. Martínez",
                "Nicolas Tagliafico": "Tagliafico",
                "Nicolás Tagliafico": "Tagliafico",
                "Rodrigo De Paul": "De Paul",
                "Rodrigo Javier De Paul": "De Paul",
                "Enzo Fernandez": "Enzo F.",
                "Enzo Fernández": "Enzo F.",
                "Alexis Mac Allister": "Mac Allister",
                "Nicolas Gonzalez": "N. González",
                "Nicolás González": "N. González",
                "Lionel Messi": "Messi",
                "Lionel Andrés Messi Cuccittini": "Messi",
                "Julian Alvarez": "Álvarez",
                "Julián Álvarez": "Álvarez",
                "Lautaro Martinez": "Lautaro",
                "Lautaro Javier Martínez": "Lautaro",
                "Exequiel Palacios": "Palacios",
                "Leandro Paredes": "Paredes",
                "Gonzalo Montiel": "Montiel",

                # Spain Starting XI & Subs
                "Unai Simon": "U. Simón",
                "Unai Simón": "U. Simón",
                "Pedro Porro": "Porro",
                "Pau Cubarsi": "Cubarsí",
                "Pau Cubarsí": "Cubarsí",
                "Aymeric Laporte": "Laporte",
                "Marc Cucurella": "Cucurella",
                "Pedri": "Pedri",
                "Pedro González López": "Pedri",
                "Rodri": "Rodri",
                "Rodrigo Hernández Cascante": "Rodri",
                "Fabian Ruiz": "Fabián",
                "Fabián Ruiz": "Fabián",
                "Fabián Ruiz Peña": "Fabián",
                "Lamine Yamal": "Lamine Yamal",
                "Lamine Yamal Nasraoui Ebana": "Lamine Yamal",
                "Dani Olmo": "Dani Olmo",
                "Daniel Olmo Carvajal": "Dani Olmo",
                "Mikel Oyarzabal": "Oyarzabal",
                "Mikel Oyarzabal Ugarte": "Oyarzabal",
                "Nico Williams": "Nico W.",
                "Nicholas Williams Arthuer": "Nico W.",
                "Ferran Torres": "Ferran",
                "Ferrán Torres García": "Ferran",
                "Martin Zubimendi": "Zubimendi",
                "Martín Zubimendi Ibáñez": "Zubimendi",
                "Alex Baena": "Baena",
                "Alejandro Baena Rodríguez": "Baena",
            }

            if p_name in KNOWN_DISPLAY_NAMES:
                short_name = KNOWN_DISPLAY_NAMES[p_name]
            else:
                parts = p_name.split()
                if len(parts) >= 2 and parts[-2].lower() in ["de", "mac", "mc", "van", "von", "di", "del", "la", "le"]:
                    short_name = f"{parts[-2]} {parts[-1]}"
                else:
                    short_name = parts[-1] if len(parts) > 1 else p_name

            # Smart adaptive vertical offset to prevent clipping off the pitch boundaries
            if py > 55:
                label_y = py - radius - 2.8
                valign = "bottom"
            else:
                label_y = py + radius + 2.8
                valign = "top"

            ax.text(
                px, label_y, short_name,
                color=self.theme["text_color"], fontsize=8.5, fontweight="bold",
                ha="center", va=valign, zorder=5,
                bbox=dict(boxstyle="round,pad=0.18", facecolor="#0b0f19", edgecolor="none", alpha=0.75)
            )

        # Tactical Title (Clean above pitch, no overlap)
        header = title or f"{team.upper()} • TACTICAL PASSING NETWORK"
        ax.set_title(header, color=self.theme["text_color"], fontsize=12, fontweight="bold", pad=12)

        hubs = network_data.get("tactical_hubs", [])
        hubs_str = ", ".join(hubs[:2]) if hubs else "N/A"
        info_text = (
            f"Network Density: {density:.3f}\n"
            f"Pass Completion: {eff.get('pass_completion_pct', 0)}%\n"
            f"Completed Passes: {eff.get('passes_completed', 0)}\n"
            f"Progressive Passes: {eff.get('progressive_passes', 0)}\n"
            f"Tactical Hubs: {hubs_str}"
        )
        # Attack Direction Indicator
        if "Spain" in team:
            ax.annotate(
                "← ATTACK DIRECTION", xy=(46.0, 4.0), xytext=(74.0, 4.0),
                arrowprops=dict(arrowstyle="->,head_width=0.3,head_length=0.4", color="#94a3b8", lw=1.2),
                color="#94a3b8", fontsize=8.0, fontweight="bold", ha="center", va="center", zorder=6
            )
        else:
            ax.annotate(
                "ATTACK DIRECTION →", xy=(74.0, 4.0), xytext=(46.0, 4.0),
                arrowprops=dict(arrowstyle="->,head_width=0.3,head_length=0.4", color="#94a3b8", lw=1.2),
                color="#94a3b8", fontsize=8.0, fontweight="bold", ha="center", va="center", zorder=6
            )

        # Position info card in lower corner with ample space
        ax.text(
            116.0, 75.0, info_text,
            color=self.theme["text_color"], fontsize=8.0,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#1e293b", edgecolor=self.theme["pitch_lines"], alpha=0.88),
            va="bottom", ha="right", zorder=6
        )

        return fig, ax

    def plot_directional_pass_map(
        self,
        df: pd.DataFrame,
        team: str,
        ax: Optional[plt.Axes] = None,
        grid_bins: Tuple[int, int] = (6, 4),
        figsize: Tuple[int, int] = (12, 8)
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Creates a directional pass flow vector map. Divides pitch into grid cells
        and computes resultant pass direction vectors.
        """
        passes = df[(df["team"] == team) & (df["type"] == "Pass")].copy()
        fig, ax = self.pitch_drawer.draw_pitch(ax=ax, figsize=figsize)

        if passes.empty:
            return fig, ax

        x_bins = np.linspace(0.0, 120.0, grid_bins[0] + 1)
        y_bins = np.linspace(0.0, 80.0, grid_bins[1] + 1)

        passes["x_bin"] = pd.cut(passes["location_x"], bins=x_bins, labels=False)
        passes["y_bin"] = pd.cut(passes["location_y"], bins=y_bins, labels=False)

        arrow_color = self.theme["argentina_primary"] if "Argentina" in team else self.theme["spain_secondary"]

        for (xb, yb), group in passes.groupby(["x_bin", "y_bin"]):
            if len(group) < 3:
                continue

            cx = (x_bins[int(xb)] + x_bins[int(xb) + 1]) / 2.0
            cy = (y_bins[int(yb)] + y_bins[int(yb) + 1]) / 2.0

            # Mean displacement
            dx = (group["end_x"] - group["location_x"]).mean()
            dy = (group["end_y"] - group["location_y"]).mean()
            norm = np.sqrt(dx**2 + dy**2) or 1.0

            # Scale arrow vector length
            arrow_scale = 8.5
            u = (dx / norm) * arrow_scale
            v = (dy / norm) * arrow_scale

            ax.annotate(
                "",
                xy=(cx + u, cy + v),
                xytext=(cx, cy),
                arrowprops=dict(
                    arrowstyle="->,head_width=0.4,head_length=0.6",
                    color=arrow_color,
                    lw=1.8,
                    mutation_scale=15
                ),
                zorder=4
            )
            # Origin dot
            ax.scatter(cx, cy, color="#ffffff", s=14, zorder=5)

        ax.set_title(
            f"{team.upper()} • DIRECTIONAL PASS FLOW VECTORS",
            color=self.theme["text_color"], fontsize=12, fontweight="bold", pad=12
        )

        return fig, ax

    def plot_progressive_passes(
        self,
        df: pd.DataFrame,
        team: str,
        ax: Optional[plt.Axes] = None,
        figsize: Tuple[int, int] = (12, 8)
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Visualizes all successful progressive and line-breaking passes entering
        the dangerous final third and penalty area.
        """
        if "is_progressive_pass" in df.columns:
            prog = df[(df["team"] == team) & (df["is_progressive_pass"] == True)].copy()
        else:
            prog = pd.DataFrame()
        fig, ax = self.pitch_drawer.draw_pitch(ax=ax, figsize=figsize)

        if prog.empty:
            ax.text(60.0, 40.0, "No progressive passes recorded", color=self.theme["text_muted"], ha="center")
            return fig, ax

        color = self.theme["goal"]

        for _, row in prog.iterrows():
            x1, y1 = row["location_x"], row["location_y"]
            x2, y2 = row["end_x"], row["end_y"]

            # Highlight passes into the box
            is_box = row.get("is_box_entry", False)
            lw = 2.4 if is_box else 1.4
            arrow_col = "#fbbf24" if is_box else color
            alpha = 0.85 if is_box else 0.55

            ax.annotate(
                "",
                xy=(x2, y2),
                xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle="->,head_width=0.35,head_length=0.5",
                    color=arrow_col,
                    lw=lw,
                    alpha=alpha,
                ),
                zorder=4
            )
            ax.scatter(x1, y1, color=arrow_col, s=12, alpha=alpha, zorder=5)

        title_str = f"{team.upper()} • PROGRESSIVE & LINE-BREAKING PASSES ({len(prog)} Total)"
        ax.set_title(
            title_str,
            color=self.theme["text_color"], fontsize=12, fontweight="bold", pad=12
        )

        return fig, ax
