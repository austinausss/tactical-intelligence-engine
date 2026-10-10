"""
Unit tests for data pipeline, coordinate cleaning, and event enrichment.
"""

import unittest
import pandas as pd
import numpy as np

from engine.pipeline.cleaner import CoordinateCleaner
from engine.pipeline.preprocessor import TacticalPreprocessor
from engine.pipeline.match_synthesizer import WorldCupFinal2026Synthesizer


class TestPipeline(unittest.TestCase):

    def setUp(self):
        self.cleaner = CoordinateCleaner()
        self.preprocessor = TacticalPreprocessor()

    def test_coordinate_cleaning_and_bounds(self):
        # Create noisy coordinate dataset
        raw_data = pd.DataFrame({
            "type": ["Pass", "Carry", "Shot"],
            "location": [[125.0, -5.0], [50.0, 40.0], np.nan],
            "pass_end_location": [[130.0, 85.0], np.nan, np.nan],
            "carry_end_location": [np.nan, [55.0, 42.0], np.nan],
            "shot_end_location": [np.nan, np.nan, [120.0, 40.0, 1.0]],
            "possession": [1, 1, 1],
            "team": ["Argentina", "Argentina", "Argentina"],
            "player": ["Lionel Messi", "Lionel Messi", "Lionel Messi"]
        })

        cleaned = self.cleaner.clean_event_coordinates(raw_data)

        # Bounds should be clamped to [0, 120] and [0, 80]
        self.assertEqual(cleaned["location_x"].iloc[0], 120.0)
        self.assertEqual(cleaned["location_y"].iloc[0], 0.0)
        self.assertEqual(cleaned["pass_end_x"].iloc[0], 120.0)
        self.assertEqual(cleaned["pass_end_y"].iloc[0], 80.0)
        self.assertFalse(cleaned["location_x"].isna().all())

    def test_progressive_pass_and_box_entry(self):
        events = pd.DataFrame({
            "type": ["Pass", "Pass", "Carry"],
            "location_x": [45.0, 85.0, 95.0],
            "location_y": [40.0, 40.0, 40.0],
            "end_x": [65.0, 105.0, 105.0],
            "end_y": [40.0, 38.0, 38.0],
            "pass_outcome": [np.nan, "Complete", np.nan],
            "team": ["Spain", "Spain", "Spain"],
            "player": ["Pedri", "Pedri", "Lamine Yamal"]
        })

        enriched = self.preprocessor.enrich_events(events)

        # Pass 1: from 45 to 65 moves 20 yds closer (progressive)
        self.assertTrue(enriched["is_progressive_pass"].iloc[0])
        # Pass 2: from 85 to 105 enters box (progressive & box entry)
        self.assertTrue(enriched["is_progressive_pass"].iloc[1])
        self.assertTrue(enriched["is_box_entry"].iloc[1])
        # Carry: from 95 to 105 enters box
        self.assertTrue(enriched["is_box_entry"].iloc[2])

    def test_synthesizer_generation(self):
        synth = WorldCupFinal2026Synthesizer(seed=123)
        events = synth.generate_full_match_events()

        self.assertGreater(len(events), 3000)
        self.assertIn("Argentina", events["team"].unique())
        self.assertIn("Spain", events["team"].unique())
        self.assertIn("Shot", events["type"].unique())
        self.assertIn("Pass", events["type"].unique())


if __name__ == "__main__":
    unittest.main()
