"""
WARNING: MasterDecisionEngine (engine.py) produces fully synthetic/random output,
not real trading decisions. See engine.py's module docstring. This script is a
report-template demo, not a live decision pipeline.
"""
import os
import pandas as pd
import openpyxl
import config
from engine import MasterDecisionEngine

def main():
    print("=== STARTING AI TRADE DECISION ENGINE (SYNTHETIC DEMO DATA - see engine.py warning) ===")
    engine = MasterDecisionEngine()
    df = engine.run_engine()
    risk_df = engine.get_capital_risk_matrix()

    # Excel Export
    excel_path = os.path.join(config.REPORTS_DIR, "AI_Trade_Decision_Model.xlsx")
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Daily Decisions', index=False)
        risk_df.to_excel(writer, sheet_name='Capital Risk Matrix', index=False)
    print(f"Excel export completed: {excel_path}")

    # Summary Text Export
    summary_path = os.path.join(config.REPORTS_DIR, "Summary.txt")
    with open(summary_path, "w") as f:
        f.write("=== AI TRADE DECISION ENGINE EXECUTIVE SUMMARY ===\n")
        f.write(f"Total Trading Days Analyzed: {len(df)}\n")
        f.write(f"Total Trade Decisions: {len(df[df['Final Recommendation']=='TRADE'])}\n")
        f.write(f"Total No-Trade Decisions: {len(df[df['Final Recommendation']=='NO TRADE'])}\n")
        f.write("\nCapital Risk Allocation Matrix:\n")
        f.write(risk_df.to_string(index=False))
    print(f"Summary text exported: {summary_path}")

    print("=== AI TRADE DECISION ENGINE COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()
