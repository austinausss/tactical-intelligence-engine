"""
Tactical Pitch Visualization Engine.
Renders broadcast-grade, dark-themed football pitches with tactical grids,
subtle turf striations, and standard StatsBomb 120 x 80 geometry.
"""

from typing import Tuple, Optional, Any
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.axes import Axes
import numpy as np

from config import PITCH_LENGTH, PITCH_WIDTH, STYLE_CONFIG


class PitchDrawer:
    """
    Renders professional football pitches with customizable themes and dimensions.
    """

    def __init__(self, theme: dict = STYLE_CONFIG):
        self.theme = theme
        self.length = PITCH_LENGTH
        self.width = PITCH_WIDTH

    def draw_pitch(
        self,
        ax: Optional[Axes] = None,
        figsize: Tuple[int, int] = (12, 8),
        half: bool = False,
        show_zones: bool = False,
        title: Optional[str] = None
    ) -> Tuple[plt.Figure, Axes]:
        """
        Draws a StatsBomb standard pitch (120x80) with clean vector graphics.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, facecolor=self.theme["bg_color"])
        else:
            fig = ax.figure

        ax.set_facecolor(self.theme["pitch_color"])

        # Determine dimensions
        x_min = 60.0 if half else 0.0
        x_max = 120.0
        y_min = 0.0
        y_max = 80.0

        # Draw subtle turf stripes
        stripe_width = 10.0
        start_x = int(x_min // stripe_width) * stripe_width
        for sx in np.arange(start_x, x_max, stripe_width * 2):
            w = min(stripe_width, x_max - sx)
            rect = patches.Rectangle(
                (sx, y_min), w, y_max - y_min,
                facecolor=self.theme["grass_stripe"],
                edgecolor="none",
                alpha=0.45,
                zorder=1
            )
            ax.add_patch(rect)

        # Tactical Thirds guides (optional)
        if show_zones:
            ax.axvline(40.0, color=self.theme["pitch_lines"], linestyle=":", alpha=0.5, zorder=2)
            ax.axvline(80.0, color=self.theme["pitch_lines"], linestyle=":", alpha=0.5, zorder=2)

        line_col = self.theme["pitch_lines"]
        lw = 1.5

        # Boundary lines
        ax.plot([x_min, x_max], [y_min, y_min], color=line_col, lw=lw, zorder=2)
        ax.plot([x_min, x_max], [y_max, y_max], color=line_col, lw=lw, zorder=2)
        ax.plot([x_max, x_max], [y_min, y_max], color=line_col, lw=lw, zorder=2)
        if not half:
            ax.plot([x_min, x_min], [y_min, y_max], color=line_col, lw=lw, zorder=2)
            # Halfway line
            ax.plot([60.0, 60.0], [y_min, y_max], color=line_col, lw=lw, zorder=2)
            # Center circle & spot
            center_circle = patches.Circle((60.0, 40.0), 10.0, edgecolor=line_col, facecolor="none", lw=lw, zorder=2)
            ax.add_patch(center_circle)
            center_spot = patches.Circle((60.0, 40.0), 0.6, color=line_col, zorder=2)
            ax.add_patch(center_spot)

        # Right Penalty Area (Defending left to right, target goal at x=120)
        # 18-yard box: (102, 18) to (120, 62)
        right_box = patches.Rectangle((102.0, 18.0), 18.0, 44.0, edgecolor=line_col, facecolor="none", lw=lw, zorder=2)
        ax.add_patch(right_box)

        # 6-yard box: (114, 30) to (120, 50)
        right_six = patches.Rectangle((114.0, 30.0), 6.0, 20.0, edgecolor=line_col, facecolor="none", lw=lw, zorder=2)
        ax.add_patch(right_six)

        # Penalty Spot & Arc (Right)
        right_pen_spot = patches.Circle((108.0, 40.0), 0.6, color=line_col, zorder=2)
        ax.add_patch(right_pen_spot)
        right_arc = patches.Arc((108.0, 40.0), 20.0, 20.0, angle=0, theta1=130, theta2=230, edgecolor=line_col, lw=lw, zorder=2)
        ax.add_patch(right_arc)

        # Right Goal
        right_goal = patches.Rectangle((120.0, 36.0), 2.5, 8.0, edgecolor=line_col, facecolor="#1e293b", lw=lw, zorder=2)
        ax.add_patch(right_goal)

        # Left Penalty Area (if full pitch)
        if not half:
            left_box = patches.Rectangle((0.0, 18.0), 18.0, 44.0, edgecolor=line_col, facecolor="none", lw=lw, zorder=2)
            ax.add_patch(left_box)
            left_six = patches.Rectangle((0.0, 30.0), 6.0, 20.0, edgecolor=line_col, facecolor="none", lw=lw, zorder=2)
            ax.add_patch(left_six)
            left_pen_spot = patches.Circle((12.0, 40.0), 0.6, color=line_col, zorder=2)
            ax.add_patch(left_pen_spot)
            left_arc = patches.Arc((12.0, 40.0), 20.0, 20.0, angle=0, theta1=310, theta2=50, edgecolor=line_col, lw=lw, zorder=2)
            ax.add_patch(left_arc)
            left_goal = patches.Rectangle((-2.5, 36.0), 2.5, 8.0, edgecolor=line_col, facecolor="#1e293b", lw=lw, zorder=2)
            ax.add_patch(left_goal)

        # Set axes limits with padding
        padding = 4.0
        ax.set_xlim(x_min - (padding if not half else 1.0), x_max + padding)
        ax.set_ylim(y_max + padding, y_min - padding)  # Invert y-axis to match StatsBomb 0 at top
        ax.set_aspect("equal")
        ax.axis("off")

        if title:
            ax.set_title(title, color=self.theme["text_color"], fontsize=14, fontweight="bold", pad=12)

        return fig, ax
