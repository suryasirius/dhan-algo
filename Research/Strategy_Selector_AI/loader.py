import os
import pandas as pd
import logging
import sys
import config

logger = logging.getLogger("StrategySelectorAI.Loader")

class DataLoader:
    def __init__(self):
        config.init_directories()

    def load_inputs(self) -> pd.DataFrame:
        if not os.path.exists(config.MARKET_DNA_PATH):
            logger.error(f"Missing Market DNA file at {config.MARKET_DNA_PATH}")
            print(f"FAIL: Market DNA file missing.")
            sys.exit(1)

        if not os.path.exists(config.MARKET_REGIMES_PATH):
            logger.error(f"Missing Market Regimes file at {config.MARKET_REGIMES_PATH}")
            print(f"FAIL: Market Regimes file missing.")
            sys.exit(1)

        logger.info("Loading Market DNA and Market Regimes datasets...")
        dna_df = pd.read_csv(config.MARKET_DNA_PATH)
        regimes_df = pd.read_csv(config.MARKET_REGIMES_PATH)

        # Standardize date keys
        dna_key = 'timestamp' if 'timestamp' in dna_df.columns else dna_df.columns[0]
        regimes_key = 'date' if 'date' in regimes_df.columns else regimes_df.columns[0]

        dna_df['clean_date'] = pd.to_datetime(dna_df[dna_key]).dt.normalize()
        regimes_df['clean_date'] = pd.to_datetime(regimes_df[regimes_key]).dt.normalize()

        # Perform inner merge on clean normalized dates
        merged_df = pd.merge(
            dna_df, 
            regimes_df, 
            left_on='clean_date', 
            right_on='clean_date', 
            how='inner',
            suffixes=('', '_regime')
        )

        # If overlap is low, fall back or align date ranges gracefully
        if len(merged_df) < 0.95 * len(dna_df):
            logger.warning(f"Merge overlap is low ({len(merged_df)}/{len(dna_df)} rows). Performing outer join alignment.")
            merged_df = pd.merge(
                dna_df, 
                regimes_df, 
                left_on='clean_date', 
                right_on='clean_date', 
                how='outer',
                suffixes=('', '_regime')
            ).dropna(subset=['clean_date'])
            merged_df = merged_df.ffill().bfill()

        logger.info(f"Successfully loaded and merged dataset. Total trading days: {len(merged_df)}")
        return merged_df
