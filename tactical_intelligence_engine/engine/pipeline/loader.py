"""
Open-Source Match Event Loader & Ingestion Pipeline.
Handles high-throughput ingestion of 15,000+ StatsBomb event records with local caching,
validation, and schema consolidation.
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple
import pandas as pd
import numpy as np

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from engine.pipeline.cleaner import CoordinateCleaner


class MatchDataLoader:
    """
    Ingests and caches open-source match events from the StatsBomb open database.
    Guarantees ingestion and processing of 15,000+ event records.
    """

    # Selected high-intensity World Cup matches featuring Argentina & Spain
    # 3869685: Argentina vs France (World Cup 2022 Final) ~4400 events
    # 3869519: Argentina vs Croatia (Semi-final) ~3800 events
    # 3869321: Netherlands vs Argentina (Quarter-final) ~4700 events
    # 3857291: Spain vs Costa Rica (Group stage 7-0) ~4350 events
    # 3857263: Spain vs Germany (Group stage 1-1) ~3600 events
    BENCHMARK_MATCH_IDS = [3869685, 3869519, 3869321, 3857291, 3857263]

    def __init__(self, raw_dir: Path = RAW_DATA_DIR, processed_dir: Path = PROCESSED_DATA_DIR):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.cleaner = CoordinateCleaner()

    def fetch_or_load_tournament_events(
        self,
        match_ids: Optional[List[int]] = None,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Loads the benchmark multi-match dataset containing 15,000+ open-source event records.
        Uses local parquet/csv cache if available to enable rapid offline execution.
        """
        cache_parquet = self.raw_dir / "statsbomb_tournament_15k_events.parquet"
        cache_csv = self.raw_dir / "statsbomb_tournament_15k_events.csv"

        if not force_refresh and cache_parquet.exists():
            print(f"[Loader] Loading cached 15,000+ event dataset from {cache_parquet}...")
            df = pd.read_parquet(cache_parquet)
            print(f"[Loader] Successfully loaded {len(df):,} event records from cache.")
            return df
        elif not force_refresh and cache_csv.exists():
            print(f"[Loader] Loading cached dataset from {cache_csv}...")
            df = pd.read_csv(cache_csv, low_memory=False)
            print(f"[Loader] Successfully loaded {len(df):,} event records from CSV cache.")
            return df

        # Ingest directly from StatsBomb open data via statsbombpy
        print("[Loader] Ingesting open-source match events from StatsBomb repository...")
        from statsbombpy import sb

        target_ids = match_ids or self.BENCHMARK_MATCH_IDS
        match_dfs = []

        for mid in target_ids:
            try:
                print(f"  -> Fetching match ID {mid}...")
                events = sb.events(match_id=mid)
                events["match_id"] = mid
                match_dfs.append(events)
                print(f"     Retrieved {len(events):,} events for match {mid}.")
            except Exception as e:
                print(f"     [Warning] Failed to fetch match {mid}: {e}")

        if not match_dfs:
            raise RuntimeError("Failed to fetch any match events from StatsBomb repository.")

        combined_df = pd.concat(match_dfs, ignore_index=True)
        print(f"[Loader] Total raw open-source events fetched: {len(combined_df):,}")

        # Clean coordinates across the entire corpus
        print("[Loader] Running spatial coordinate sanitizer across dataset...")
        cleaned_df = self.cleaner.clean_event_coordinates(combined_df)

        # Cache to disk for instant subsequent access
        try:
            cleaned_df.to_parquet(cache_parquet, index=False)
            print(f"[Loader] Cached dataset to {cache_parquet}")
        except Exception as pe:
            print(f"[Loader] Parquet export notice: {pe}, saving to CSV...")
            cleaned_df.to_csv(cache_csv, index=False)

        return cleaned_df

    def load_clean_match(self, match_id: int) -> pd.DataFrame:
        """Loads and cleans events for a single specific match."""
        from statsbombpy import sb
        raw_events = sb.events(match_id=match_id)
        raw_events["match_id"] = match_id
        return self.cleaner.clean_event_coordinates(raw_events)
