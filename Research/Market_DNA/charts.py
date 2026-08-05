import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import config

logger = logging.getLogger("MarketDNA.Charts")

class ChartGenerator:
    def __init__(self, dna_df: pd.DataFrame):
        self.dna_df = dna_df.copy()
        config.init_directories()
        os.makedirs(config.CHARTS_DIR, exist_ok=True)
        sns.set_theme(style="whitegrid")

    def generate_all_charts(self):
        """Generates all institutional research charts."""
        logger.info("Generating institutional research charts...")
        
        if self.dna_df.empty:
            logger.warning("DataFrame is empty. Skipping chart generation.")
            return

        self._plot_distribution('Distance_from_VWAP', 'Distance from VWAP (%)', 'vwap_distribution.png')
        self._plot_categorical_frequency('Market_Type', 'Market Type Frequency', 'market_type_frequency.png', rot=45)
        self._plot_categorical_frequency('Previous_Day_Trend', 'Trend Distribution', 'trend_distribution.png')
        self._plot_distribution('Opening_Volume', 'Opening Volume Distribution', 'volume_distribution.png')

        logger.info(f"All charts successfully generated and saved to {config.CHARTS_DIR}")

    def _plot_distribution(self, column: str, title: str, filename: str):
        if column not in self.dna_df.columns:
            return
        plt.figure(figsize=(8, 5))
        sns.histplot(self.dna_df[column].dropna(), kde=True, color='royalblue', bins=30)
        plt.title(title, fontsize=12, fontweight='bold')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(config.CHARTS_DIR, filename), dpi=300)
        plt.close()

    def _plot_categorical_frequency(self, column: str, title: str, filename: str, rot=0):
        if column not in self.dna_df.columns:
            return
        plt.figure(figsize=(8, 5))
        ax = self.dna_df[column].value_counts().plot(kind='bar', color='darkslategray', edgecolor='black')
        plt.title(title, fontsize=12, fontweight='bold')
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.xticks(rotation=rot)
        plt.tight_layout()
        plt.savefig(os.path.join(config.CHARTS_DIR, filename), dpi=300)
        plt.close()
