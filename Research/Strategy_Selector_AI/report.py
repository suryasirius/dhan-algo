import os
import pandas as pd
import config
import logging

logger = logging.getLogger("StrategySelectorAI.Report")

class InstitutionalReporter:
    def __init__(self, df: pd.DataFrame, recommendations_df: pd.DataFrame):
        self.df = df
        self.recommendations = recommendations_df
        config.init_directories()

    def export_all_reports(self):
        logger.info("Exporting multi-format institutional reports...")
        
        csv_path = os.path.join(config.REPORTS_DIR, "Strategy_Selector_Output.csv")
        excel_path = os.path.join(config.REPORTS_DIR, "Strategy_Selector_Output.xlsx")
        html_path = os.path.join(config.REPORTS_DIR, "Strategy_Selector_Report.html")
        
        self.recommendations.to_csv(csv_path, index=False)
        self.recommendations.to_excel(excel_path, index=False)
        
        html_content = f"""
        <html>
        <head>
            <title>Strategy Selector AI - Institutional Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f8f9fa; }}
                h1 {{ color: #1f2937; }}
                table {{ border-collapse: collapse; width: 100%; background: white; }}
                th, td {{ border: 1px solid #dee2e6; padding: 10px; text-align: left; }}
                th {{ background-color: #343a40; color: white; }}
            </style>
        </head>
        <body>
            <h1>Strategy Selector AI - Institutional Execution Report</h1>
            <p>Generated for Quantitative Research Platform.</p>
            {self.recommendations.head(50).to_html(index=False)}
        </body>
        </html>
        """
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
            
        logger.info(f"Reports successfully exported to {config.REPORTS_DIR}")
