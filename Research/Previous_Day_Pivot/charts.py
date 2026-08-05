import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, Any
from matplotlib.backends.backend_pdf import PdfPages
import config

class ChartGenerator:
    def __init__(self, metrics_df: pd.DataFrame, stats_results: Dict[str, Any], logger: logging.Logger):
        self.df = metrics_df.copy()
        self.stats = stats_results
        self.logger = logger
        
        # Ensure index is datetime for grouping
        if not isinstance(self.df.index, pd.DatetimeIndex):
            self.df.index = pd.to_datetime(self.df.index)
            
        self.df['Month'] = self.df.index.month
        self.df['Weekday'] = self.df.index.weekday

    def generate_all_charts(self) -> None:
        self.logger.info("Generating professional analytical charts and PDF report...")
        sns.set_theme(style="whitegrid")
        
        pdf_path = config.REPORTS_DIR / 'Pivot_Research_Visualizations.pdf'
        
        with PdfPages(pdf_path) as pdf:
            # 1. Histogram of Moves
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(self.df['Max_Move_Above_Pivot'], kde=True, color="forestgreen", label="Max Move Above Pivot", ax=ax, alpha=0.6)
            sns.histplot(self.df['Max_Move_Below_Pivot'], kde=True, color="crimson", label="Max Move Below Pivot", ax=ax, alpha=0.6)
            ax.set_title("Distribution of Maximum Intraday Price Extensions Around Pivot", fontsize=12, fontweight='bold')
            ax.set_xlabel("Price Distance", fontsize=10)
            ax.set_ylabel("Frequency", fontsize=10)
            ax.legend()
            plt.tight_layout()
            plt.savefig(config.CHARTS_DIR / 'histogram_moves.png', dpi=300)
            pdf.savefig(fig)
            plt.close(fig)

            # 2. Scatter Plot: Trend Strength vs Time Above Pivot
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(
                x='Trend_Strength', 
                y='Time_Above_Pivot', 
                hue='Intraday_Trend', 
                data=self.df, 
                palette={'Bullish': 'forestgreen', 'Bearish': 'crimson', 'Neutral': 'gray'},
                alpha=0.8,
                ax=ax
            )
            ax.set_title("Intraday Trend Strength vs Time Spent Above Pivot", fontsize=12, fontweight='bold')
            ax.set_xlabel("Trend Strength Ratio", fontsize=10)
            ax.set_ylabel("Time Above Pivot (Minutes)", fontsize=10)
            plt.tight_layout()
            plt.savefig(config.CHARTS_DIR / 'scatter_trend.png', dpi=300)
            pdf.savefig(fig)
            plt.close(fig)

            # 3. Heatmap: Monthly Comparison
            monthly_acc = self.df.groupby('Month')['Pivot_Predicts'].mean().to_frame(name='Accuracy')
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(monthly_acc, annot=True, cmap="YlGnBu", fmt=".1%", cbar=True, ax=ax)
            ax.set_title("Monthly Pivot Trend Prediction Accuracy Heatmap", fontsize=12, fontweight='bold')
            ax.set_ylabel("Month Index", fontsize=10)
            plt.tight_layout()
            plt.savefig(config.CHARTS_DIR / 'heatmap_monthly.png', dpi=300)
            pdf.savefig(fig)
            plt.close(fig)

            # 4. Weekly Comparison
            weekly_acc = self.df.groupby('Weekday')['Pivot_Predicts'].mean().to_frame(name='Accuracy')
            weekday_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            weekly_acc.index = [weekday_names[i] for i in weekly_acc.index if i < len(weekday_names)]
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.barplot(x=weekly_acc.index, y='Accuracy', data=weekly_acc, palette="viridis", ax=ax)
            ax.set_title("Day-of-Week Trend Prediction Accuracy", fontsize=12, fontweight='bold')
            ax.set_xlabel("Day of Week", fontsize=10)
            ax.set_ylabel("Prediction Accuracy Rate", fontsize=10)
            ax.set_ylim(0, 1.0)
            plt.tight_layout()
            plt.savefig(config.CHARTS_DIR / 'weekly_comparison.png', dpi=300)
            pdf.savefig(fig)
            plt.close(fig)

            # 5. Distribution of Pivot Touches
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.countplot(x='Num_Pivot_Touches', data=self.df, palette="crest", ax=ax)
            ax.set_title("Frequency Distribution of Daily Intraday Pivot Touches", fontsize=12, fontweight='bold')
            ax.set_xlabel("Number of Pivot Crosses/Touches", fontsize=10)
            ax.set_ylabel("Number of Days", fontsize=10)
            plt.tight_layout()
            plt.savefig(config.CHARTS_DIR / 'distribution_touches.png', dpi=300)
            pdf.savefig(fig)
            plt.close(fig)

        self.logger.info(f"Charts successfully generated and saved to {config.CHARTS_DIR} and PDF report compiled.")