import os

files = {
    "config.py": '''import os

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_PATH = os.path.join(DATA_DIR, "market_data.csv")

# Directories and Aliases
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
OUTPUT_DIR = OUTPUTS_DIR
CHARTS_DIR = os.path.join(OUTPUTS_DIR, "charts")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

def init_directories():
    """Initializes all necessary project directories."""
    for path in [DATA_DIR, OUTPUTS_DIR, CHARTS_DIR, LOGS_DIR]:
        os.makedirs(path, exist_ok=True)
''',

    "utils.py": '''import logging
import os
import config

def setup_logger():
    """Sets up unified logging for the Market DNA platform."""
    config.init_directories()
    log_file = os.path.join(config.LOGS_DIR, "market_dna.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("MarketDNA")
''',

    "loader.py": '''import os
import pandas as pd
import numpy as np
import logging
import config

logger = logging.getLogger("MarketDNA.Loader")

class DataLoader:
    def __init__(self):
        config.init_directories()

    def load_data(self) -> pd.DataFrame:
        """Loads intraday OHLCV data or generates robust synthetic data if missing."""
        if os.path.exists(config.DATA_PATH):
            logger.info(f"Loading market data from {config.DATA_PATH}")
            df = pd.read_csv(config.DATA_PATH)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        else:
            logger.warning(f"Data file not found at {config.DATA_PATH}. Generating synthetic intraday dataset...")
            return self._generate_synthetic_data()

    def _generate_synthetic_data(self) -> pd.DataFrame:
        """Generates realistic 1-minute OHLCV sample data for testing."""
        np.random.seed(42)
        date_range = pd.date_range(start="2026-01-01 09:15:00", end="2026-03-31 15:30:00", freq="min")
        
        date_range = date_range[
            (date_range.time >= pd.to_datetime("09:15:00").time()) & 
            (date_range.time <= pd.to_datetime("15:30:00").time())
        ]
        
        n = len(date_range)
        returns = np.random.normal(loc=0.00001, scale=0.001, size=n)
        price_path = 20000 * np.cumprod(1 + returns)
        
        df = pd.DataFrame({
            'timestamp': date_range,
            'open': price_path * (1 + np.random.uniform(-0.0005, 0.0005, n)),
            'high': price_path * (1 + np.random.uniform(0.0001, 0.0015, n)),
            'low': price_path * (1 - np.random.uniform(0.0001, 0.0015, n)),
            'close': price_path,
            'volume': np.random.randint(1000, 50000, size=n)
        })
        
        df.to_csv(config.DATA_PATH, index=False)
        logger.info(f"Synthetic dataset successfully generated and saved to {config.DATA_PATH}")
        return df
''',

    "feature_engine.py": '''import pandas as pd
import numpy as np

class FeatureEngine:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def extract_all_features(self) -> pd.DataFrame:
        """Extracts quantitative DNA features and provides all required naming aliases."""
        if 'timestamp' not in self.df.columns:
            raise ValueError("Dataset must contain a 'timestamp' column.")

        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.df = self.df.sort_values('timestamp').reset_index(drop=True)
        
        self.df['date'] = self.df['timestamp'].dt.date
        self.df['range'] = self.df['high'] - self.df['low']
        
        daily_groups = self.df.groupby('date')
        dna_records = []
        prev_close = None
        prev_trend = "Neutral"
        
        for date, group in daily_groups:
            open_price = group.iloc[0]['open']
            high_price = group['high'].max()
            low_price = group['low'].min()
            close_price = group.iloc[-1]['close']
            total_volume = group['volume'].sum()
            opening_volume = group.iloc[0]['volume']
            
            gap_pct = ((open_price - prev_close) / prev_close) * 100 if (prev_close is not None and prev_close > 0) else 0.0
                
            typical_price = (group['high'] + group['low'] + group['close']) / 3
            vwap = (typical_price * group['volume']).sum() / total_volume if total_volume > 0 else close_price
            vwap_distance = ((close_price - vwap) / vwap) * 100
            
            atr = group['range'].mean()
            pct_chg = ((close_price - open_price) / open_price) * 100
            current_trend = "Bullish" if pct_chg > 0 else ("Bearish" if pct_chg < 0 else "Neutral")
            
            dna_records.append({
                'date': date,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': total_volume,
                'Opening_Volume': opening_volume,
                'net_change': close_price - open_price,
                'pct_change': pct_chg,
                'day_range': high_price - low_price,
                'Gap_Pct': gap_pct,
                'VWAP_Distance': vwap_distance,
                'Distance_from_VWAP': vwap_distance,
                'ATR': atr,
                'Previous_Day_Trend': prev_trend
            })
            
            prev_close = close_price
            prev_trend = current_trend
            
        return pd.DataFrame(dna_records)
''',

    "classifier.py": '''import pandas as pd
import numpy as np

class MarketClassifier:
    def __init__(self, dna_df: pd.DataFrame):
        self.df = dna_df.copy()

    def classify_sessions(self) -> pd.DataFrame:
        """Classifies trading sessions into institutional archetypes with full aliases."""
        if self.df.empty:
            return self.df

        archetypes = []
        for _, row in self.df.iterrows():
            pct = row.get('pct_change', 0)
            if pct > 0.5:
                archetype = "Strong Trend Up"
            elif pct < -0.5:
                archetype = "Strong Trend Down"
            elif abs(pct) <= 0.1:
                archetype = "Inside Day / Range"
            else:
                archetype = "Moderate Trend"
            archetypes.append(archetype)

        self.df['market_archetype'] = archetypes
        self.df['Market_Type'] = archetypes
        self.df['Regime'] = archetypes
        
        return self.df
''',

    "charts.py": '''import os
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
''',

    "report.py": '''import os
import pandas as pd
import logging
import config

logger = logging.getLogger("MarketDNA.Report")

class InstitutionalReporter:
    def __init__(self, dna_df: pd.DataFrame):
        self.df = dna_df.copy()
        config.init_directories()
        
    def export_all_reports(self):
        """Exports multi-format institutional quantitative research reports."""
        logger.info("Exporting multi-format institutional reports...")
        
        csv_path = os.path.join(config.OUTPUTS_DIR, "Market_DNA.csv")
        self.df.to_csv(csv_path, index=False)
        logger.info(f"Successfully exported CSV report to {csv_path}")
        
        if not self.df.empty:
            md_path = os.path.join(config.OUTPUTS_DIR, "Market_DNA_Summary.md")
            summary_stats = self.df.describe().to_markdown()
            
            md_content = f"""# Market DNA Quantitative Research Report
Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview Summary Statistics
{summary_stats}
"""
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md_content)
            logger.info(f"Successfully exported Markdown report to {md_path}")
''',

    "run.py": '''import config
from utils import setup_logger
from loader import DataLoader
from feature_engine import FeatureEngine
from classifier import MarketClassifier
from charts import ChartGenerator
from report import InstitutionalReporter

def main():
    logger = setup_logger()
    logger.info("=== INITIALIZING MARKET DNA QUANTITATIVE PLATFORM ===")
    
    loader = DataLoader()
    raw_df = loader.load_data()
    
    feature_engine = FeatureEngine(raw_df)
    dna_df = feature_engine.extract_all_features()
    
    classifier = MarketClassifier(dna_df)
    dna_df = classifier.classify_sessions()
    
    chart_gen = ChartGenerator(dna_df)
    chart_gen.generate_all_charts()
    
    reporter = InstitutionalReporter(dna_df)
    reporter.export_all_reports()
    
    logger.info("=== MARKET DNA PIPELINE COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()
'''
}

for filename, content in files.items():
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Successfully created: {filename}")

print("\nAll project files generated successfully! You can now run:")
print("python run.py")