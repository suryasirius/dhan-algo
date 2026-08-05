import os
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
