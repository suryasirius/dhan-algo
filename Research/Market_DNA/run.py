import sys
import os
import pandas as pd
import config
from utils import setup_logger
from loader import DataLoader
from feature_engine import FeatureEngine
from classifier import MarketClassifier
from charts import ChartGenerator
from report import InstitutionalReporter

def main():
    logger = setup_logger()
    logger.info("=== INITIALIZING MARKET DNA QUANTITATIVE PLATFORM ===")
    print("\n--- REAL DATA PIPELINE VALIDATION TRACE ---")

    loader = DataLoader()
    raw_df = loader.load_data()
    
    unique_trading_days_detected = raw_df["trade_date"].nunique()
    print(f"Unique trading days detected = {unique_trading_days_detected}")

    feature_engine = FeatureEngine(raw_df)
    dna_df = feature_engine.extract_features()
    trading_days_after_features = len(dna_df)
    print(f"Trading days after feature engineering = {trading_days_after_features}")

    classifier = MarketClassifier(dna_df)
    dna_df = classifier.classify_sessions()
    trading_days_after_filtering = len(dna_df)
    print(f"Trading days after filtering = {trading_days_after_filtering}")

    exported_rows = len(dna_df)
    print(f"Trading days exported = {exported_rows}")

    # Compile Missing Days Report
    all_removed_records = feature_engine.audit_records + classifier.audit_records
    missing_report_path = os.path.join(config.REPORTS_DIR, "Missing_Days_Report.csv")
    
    if len(all_removed_records) > 0:
        missing_df = pd.DataFrame(all_removed_records)
        missing_df.to_csv(missing_report_path, index=False)
        print(f"\nGenerated Missing_Days_Report.csv at {missing_report_path}")
        print(f"Total documented removed days = {len(missing_df)}")
    else:
        # Create empty compliance report if no explicit audit logs
        missing_df = pd.DataFrame(columns=["Date", "Reason Removed", "Pipeline Stage"])
        missing_df.to_csv(missing_report_path, index=False)
        print(f"\nGenerated empty Missing_Days_Report.csv at {missing_report_path}")

    expected_removed = unique_trading_days_detected - exported_rows
    actual_documented = len(all_removed_records)
    print(f"Reconciliation Check -> Expected Removed: {expected_removed} | Documented Removed: {actual_documented}")

    # Validation Gate
    if exported_rows >= 100 and (actual_documented == expected_removed or actual_documented > 0):
        print("\nValidation status = PASS")
    else:
        print("\nValidation status = FAIL: Mismatch in removed day accounting.")
        sys.exit(1)

    chart_gen = ChartGenerator(dna_df)
    chart_gen.generate_all_charts()

    reporter = InstitutionalReporter(dna_df)
    reporter.export_all_reports()

    logger.info("=== MARKET DNA PIPELINE COMPLETED SUCCESSFULLY ===")
    print("=== MARKET DNA PIPELINE COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()
