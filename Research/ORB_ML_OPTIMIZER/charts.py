import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, Any
from matplotlib.backends.backend_pdf import PdfPages
import config

class ChartGenerator:
    def __init__(self, trades_df: pd.DataFrame, stats_results: Dict[str, Any], logger: logging.Logger):
        self.trades = trades_df.copy()
        self.stats = stats_results
        self.logger = logger

    def generate_all_charts(self) -> None:
        self.logger.info("Generating institutional research charts, heatmaps, and PDF report...")
        sns.set_theme(style="whitegrid")
        
        pdf_path = config.REPORTS_DIR / 'ORB_ML_Optimizer_Visualizations.pdf'
        
        with PdfPages(pdf_path) as pdf:
            if not self.trades.empty:
                # 1. Profit Curve
                fig, ax = plt.subplots(figsize=(10, 6))
                self.trades['cumulative_profit'] = self.trades['net_pnl'].cumsum()
                sns.lineplot(x=range(len(self.trades)), y='cumulative_profit', data=self.trades, color="forestgreen", ax=ax, linewidth=2)
                ax.set_title("Strategy Cumulative Net Profit Curve", fontsize=12, fontweight='bold')
                ax.set_xlabel("Trade Index", fontsize=10)
                ax.set_ylabel("Net Profit (INR)", fontsize=10)
                plt.tight_layout()
                plt.savefig(config.CHARTS_DIR / 'profit_curve.png', dpi=300)
                pdf.savefig(fig)
                plt.close(fig)

                # 2. Drawdown Curve
                fig, ax = plt.subplots(figsize=(10, 6))
                self.trades['equity'] = config.INITIAL_CAPITAL + self.trades['net_pnl'].cumsum()
                self.trades['peak'] = self.trades['equity'].cummax()
                self.trades['drawdown'] = self.trades['equity'] - self.trades['peak']
                sns.lineplot(x=range(len(self.trades)), y='drawdown', data=self.trades, color="crimson", ax=ax, linewidth=2)
                ax.set_title("Strategy Drawdown Curve", fontsize=12, fontweight='bold')
                ax.set_xlabel("Trade Index", fontsize=10)
                ax.set_ylabel("Drawdown (INR)", fontsize=10)
                plt.tight_layout()
                plt.savefig(config.CHARTS_DIR / 'drawdown_curve.png', dpi=300)
                pdf.savefig(fig)
                plt.close(fig)

                # 3. Correlation Matrix Heatmap
                fig, ax = plt.subplots(figsize=(8, 6))
                corr = self.trades[['orb_range', 'gap_pct', 'net_pnl', 'win', 'atr', 'volume']].corr()
                sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", cbar=True, ax=ax)
                ax.set_title("Feature & Performance Correlation Matrix", fontsize=12, fontweight='bold')
                plt.tight_layout()
                plt.savefig(config.CHARTS_DIR / 'correlation_matrix.png', dpi=300)
                pdf.savefig(fig)
                plt.close(fig)

        self.logger.info(f"Charts successfully generated and saved to {config.CHARTS_DIR} and PDF report compiled.")