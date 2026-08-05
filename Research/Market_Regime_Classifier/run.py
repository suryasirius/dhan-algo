import sys
import config
from utils import setup_logger
from loader import DataLoader
from classifier import RegimeAutoLabeler
from ml_models import RegimeMLTrainer
from charts import RegimeChartGenerator
from report import RegimeReporter

def main():
    logger = setup_logger()
    logger.info("=== INITIALIZING MARKET REGIME CLASSIFIER ===")
    print("\\n--- PIPELINE STAGE VALIDATION TRACE ---")
    
    # 1. Load Data
    loader = DataLoader()
    raw_df = loader.load_data()
    
    # 2. Auto-label Regimes & Compute Scores
    labeler = RegimeAutoLabeler(raw_df)
    labeled_df = labeler.compute_scores_and_labels()
    
    # 3. Validate & Export (Stops before ML if validation fails)
    reporter = RegimeReporter(labeled_df, {})
    validation_passed = reporter.validate_and_export()
    
    if not validation_passed:
        print("\\n[STOP] Pipeline halted due to row-count validation failure.")
        print("FAIL: Check reports/Validation_Report.txt for details.")
        sys.exit(1)
        
    print("Validation PASSED. Proceeding to Machine Learning training...")
    
    # 4. Train & Evaluate ML Models
    trainer = RegimeMLTrainer(labeled_df)
    ml_results = trainer.train_and_evaluate()
    
    # Update reporter with results for final HTML reporting
    reporter.results = ml_results
    
    # 5. Generate Charts
    chart_gen = RegimeChartGenerator(labeled_df, ml_results)
    chart_gen.generate_all_charts()
    
    logger.info("=== PIPELINE COMPLETED SUCCESSFULLY ===")
    print("\\n=== PIPELINE COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()

