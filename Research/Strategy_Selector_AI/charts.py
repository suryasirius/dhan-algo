import matplotlib.pyplot as plt
import seaborn as sns
import os
import config
import logging

logger = logging.getLogger("StrategySelectorAI.Charts")

class ChartGenerator:
    def __init__(self, df: pd.DataFrame, recommendations_df: pd.DataFrame):
        self.df = df
        self.recommendations = recommendations_df
        config.init_directories()

    def generate_all_charts(self):
        logger.info("Generating institutional research charts...")
        sns.set_theme(style="whitegrid")
        
        plt.figure(figsize=(8, 5))
        sns.countplot(data=self.recommendations, x="Best Strategy", palette="viridis")
        plt.title("Strategy Distribution Across Trading Days")
        plt.savefig(os.path.join(config.CHARTS_DIR, "Strategy_Distribution.png"))
        plt.close()
        
        plt.figure(figsize=(8, 5))
        sns.histplot(self.recommendations["Confidence"], kde=True, color="teal")
        plt.title("Model Confidence Distribution")
        plt.savefig(os.path.join(config.CHARTS_DIR, "Confidence_Distribution.png"))
        plt.close()
        
        plt.figure(figsize=(8, 6))
        numeric_cols = self.df.select_dtypes(include=["float64", "int64"]).columns[:6]
        corr = self.df[numeric_cols].corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Feature Correlation Matrix")
        plt.savefig(os.path.join(config.CHARTS_DIR, "Correlation_Matrix.png"))
        plt.close()
        
        logger.info("All charts successfully generated and saved.")
