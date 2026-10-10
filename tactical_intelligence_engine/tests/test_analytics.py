"""
Unit tests for analytics components: xG, xG Chain, Passing Networks, Field Tilt, and Tactical Structures.
"""

import unittest
import pandas as pd
import numpy as np

from engine.analytics.xg_engine import XGEngine
from engine.analytics.xg_chain import XGChainEngine
from engine.analytics.passing_network import PassingNetworkEngine
from engine.analytics.player_influence import PlayerInfluenceEngine
from engine.analytics.tactical_structure import TacticalStructureEngine


class TestAnalytics(unittest.TestCase):

    def setUp(self):
        self.xg_engine = XGEngine()
        self.chain_engine = XGChainEngine()
        self.pass_engine = PassingNetworkEngine(min_pass_threshold=1)
        self.influence_engine = PlayerInfluenceEngine()
        self.structure_engine = TacticalStructureEngine()

    def test_xg_engine_calibration_and_efficiency(self):
        shots = pd.DataFrame({
            "type": ["Shot", "Shot"],
            "location_x": [112.0, 90.0],
            "location_y": [40.0, 40.0],
            "shot_outcome": ["Goal", "Saved"],
            "team": ["Argentina", "Argentina"],
            "player": ["Lionel Messi", "Julián Álvarez"]
        })

        calibrated = self.xg_engine.calibrate_or_fill_xg(shots)
        self.assertIn("xg", calibrated.columns)
        # Closer shot should have higher xG
        self.assertGreater(calibrated["xg"].iloc[0], calibrated["xg"].iloc[1])

        eff = self.xg_engine.compute_team_shot_efficiency(calibrated, "Argentina")
        self.assertEqual(eff["goals"], 1)
        self.assertEqual(eff["total_shots"], 2)
        self.assertGreater(eff["xg_efficiency_ratio"], 0.0)

    def test_xg_chain_calculation(self):
        chain_data = pd.DataFrame({
            "match_id": [1, 1, 1, 1],
            "period": [1, 1, 1, 1],
            "possession": [1, 1, 1, 1],
            "type": ["Pass", "Pass", "Pass", "Shot"],
            "player": ["Enzo Fernández", "Alexis Mac Allister", "Lionel Messi", "Julián Álvarez"],
            "pass_recipient": ["Alexis Mac Allister", "Lionel Messi", "Julián Álvarez", np.nan],
            "xg": [0.0, 0.0, 0.0, 0.45],
            "shot_outcome": [np.nan, np.nan, np.nan, "Goal"],
            "team": ["Argentina", "Argentina", "Argentina", "Argentina"]
        })

        chain_res = self.chain_engine.compute_chain_metrics(chain_data)
        
        # All 4 players participated in chain, so all should have xg_chain = 0.45
        self.assertEqual(len(chain_res), 4)
        for _, row in chain_res.iterrows():
            self.assertAlmostEqual(row["xg_chain"], 0.45, places=2)

        # Enzo and Mac Allister are in buildup (not shooter Álvarez, not assister Messi)
        enzo_row = chain_res[chain_res["player"] == "Enzo Fernández"].iloc[0]
        self.assertAlmostEqual(enzo_row["xg_buildup"], 0.45, places=2)

    def test_passing_network_topology(self):
        pass_data = pd.DataFrame({
            "type": ["Pass", "Pass", "Pass"],
            "team": ["Spain", "Spain", "Spain"],
            "player": ["Rodri", "Pedri", "Rodri"],
            "pass_recipient": ["Pedri", "Lamine Yamal", "Pedri"],
            "pass_outcome": [np.nan, np.nan, np.nan],
            "location_x": [60.0, 75.0, 62.0],
            "location_y": [40.0, 48.0, 41.0],
            "end_x": [75.0, 95.0, 75.0],
            "end_y": [48.0, 20.0, 48.0],
            "minute": [10, 10, 12]
        })

        net = self.pass_engine.compute_team_network(pass_data, "Spain")
        self.assertGreater(net["network_density"], 0.0)
        self.assertEqual(net["total_completed_passes"], 3)
        self.assertEqual(len(net["nodes"]), 3)

    def test_field_tilt_and_influence(self):
        events = pd.DataFrame({
            "team": ["Argentina", "Argentina", "Spain"],
            "player": ["Lionel Messi", "Rodrigo De Paul", "Pedri"],
            "position": ["Forward", "Midfielder", "Midfielder"],
            "type": ["Pass", "Pass", "Pass"],
            "location_x": [85.0, 92.0, 45.0],  # 2 in final third for Arg, 0 for Spain
            "location_y": [40.0, 50.0, 40.0],
            "end_x": [90.0, 95.0, 55.0],
            "end_y": [42.0, 52.0, 42.0],
            "is_progressive_pass": [True, False, False],
            "is_progressive_carry": [False, False, False],
            "is_box_entry": [False, False, False],
            "is_defensive_action": [False, False, False],
            "is_high_press": [False, False, False]
        })

        tilt = self.influence_engine.compute_match_field_tilt(events, "Argentina", "Spain")
        self.assertEqual(tilt["field_tilt_a"], 100.0)
        self.assertEqual(tilt["field_tilt_b"], 0.0)


if __name__ == "__main__":
    unittest.main()
