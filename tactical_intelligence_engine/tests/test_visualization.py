"""
Unit tests for visualization components: pitch drawing, heatmaps, pass maps, and shot charts.
"""

import unittest
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for testing
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from engine.visualization.pitch import PitchDrawer
from engine.visualization.heatmaps import HeatmapVisualizer
from engine.visualization.pass_maps import PassMapVisualizer
from engine.visualization.shot_charts import ShotChartVisualizer


class TestVisualization(unittest.TestCase):

    def setUp(self):
        self.pitch_drawer = PitchDrawer()
        self.heatmap_viz = HeatmapVisualizer()
        self.pass_viz = PassMapVisualizer()
        self.shot_viz = ShotChartVisualizer()

    def test_draw_pitch_full_and_half(self):
        fig1, ax1 = self.pitch_drawer.draw_pitch(half=False)
        self.assertIsNotNone(fig1)
        self.assertIsNotNone(ax1)
        plt.close(fig1)

        fig2, ax2 = self.pitch_drawer.draw_pitch(half=True)
        self.assertIsNotNone(fig2)
        self.assertIsNotNone(ax2)
        plt.close(fig2)

    def test_player_heatmap(self):
        data = pd.DataFrame({
            "player": ["Lionel Messi"] * 10,
            "team": ["Argentina"] * 10,
            "position": ["Forward"] * 10,
            "location_x": np.linspace(70, 95, 10),
            "location_y": np.linspace(35, 55, 10),
        })

        fig, ax = self.heatmap_viz.plot_player_heatmap(data, "Lionel Messi")
        self.assertIsNotNone(fig)
        plt.close(fig)

    def test_shot_chart(self):
        shots = pd.DataFrame({
            "player": ["Lionel Messi", "Julián Álvarez"],
            "team": ["Argentina", "Argentina"],
            "type": ["Shot", "Shot"],
            "location_x": [110.0, 105.0],
            "location_y": [40.0, 35.0],
            "xg": [0.45, 0.22],
            "shot_outcome": ["Goal", "Saved"],
            "minute": [41, 65]
        })

        fig, ax = self.shot_viz.plot_shot_chart(shots, "Argentina")
        self.assertIsNotNone(fig)
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
