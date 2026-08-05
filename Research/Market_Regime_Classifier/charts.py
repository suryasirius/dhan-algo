import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import config
logger = logging.getLogger("MarketRegime.Charts")
class RegimeChartGenerator:
    def __init__(self, df: pd.DataFrame, ml_results: dict):
        self.df = df.copy()
        self.results = ml_results
        config.init_directories()
        sns.set_theme(style="whitegrid")
    def generate_all_charts(self):
        logger.info("Generating charts...")
        plt.figure(figsize=(10, 6))
        self.df["Market_Type"].value_counts().plot(kind="bar", color="teal", edgecolor="black")
        plt.title("Market Regime Distribution", fontsize=14, fontweight="bold")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(config.CHARTS_DIR, "regime_distribution.png"), dpi=300)
        plt.close()
        
        metrics_data = [{"Model": k, "Metric": "Accuracy", "Score": v["Accuracy"]} for k, v in self.results.items()]
        metrics_data += [{"Model": k, "Metric": "F1 Score", "Score": v["F1 Score"]} for k, v in self.results.items()]
        plt.figure(figsize=(10, 6))
        sns.barplot(data=pd.DataFrame(metrics_data), x="Model", y="Score", hue="Metric", palette="viridis")
        plt.title("ML Models Comparative Performance", fontsize=14, fontweight="bold")
        plt.ylim(0, 1.05)
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(os.path.join(config.CHARTS_DIR, "ml_model_comparison.png"), dpi=300)
        plt.close()

