import os
import pandas as pd
import logging
import config

logger = logging.getLogger("MarketRegime.Report")

class RegimeReporter:
    def __init__(self, df: pd.DataFrame, ml_results: dict):
        self.df = df.copy()
        self.results = ml_results
        config.init_directories()

    def validate_and_export(self) -> bool:
        row_count = len(self.df)
        print(f"Rows before export = {row_count}")
        
        # Validation Check: Expecting at least 20 trading days to consider dataset valid (or adjust based on input)
        EXPECTED_MIN_DAYS = 10 
        
        if row_count < EXPECTED_MIN_DAYS:
            print(f"FAIL: Row count ({row_count}) is below expected minimum trading days ({EXPECTED_MIN_DAYS}). Data pipeline loss detected.")
            logger.error(f"FAIL: Row count {row_count} is too low. Stopping execution.")
            
            # Generate Validation_Report.txt
            val_report_path = os.path.join(config.REPORTS_DIR, "Validation_Report.txt")
            with open(val_report_path, "w", encoding="utf-8") as f:
                f.write("STATUS: FAIL\\n")
                f.write(f"Reason: Exported rows ({row_count}) are less than expected trading days threshold.\\n")
                f.write("Root Cause: Input dataset Market_DNA.csv contains collapsed or insufficient rows.\\n")
            print(f"Generated Validation_Report.txt at {val_report_path}")
            return False
            
        # Export files
        csv_path = os.path.join(config.OUTPUT_DIR, "Market_Regimes.csv")
        self.df.to_csv(csv_path, index=False)
        print(f"Rows exported = {len(self.df)}")
        logger.info(f"Successfully exported {len(self.df)} rows to {csv_path}")
        
        excel_path = os.path.join(config.OUTPUT_DIR, "Market_Regimes.xlsx")
        self.df.to_excel(excel_path, index=False)
        
        html_path = os.path.join(config.REPORTS_DIR, "Market_Regime_Report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write("<h1>Market Regime Classification Report</h1>" + self.df.tail(15).to_html())
            
        return True

