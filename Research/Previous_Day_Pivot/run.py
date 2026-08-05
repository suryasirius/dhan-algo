import sys
import logging
import traceback
from config import PROJECT_ROOT

# Assuming all modules will be available in the same directory
try:
    import utils
    from pivot_engine import PivotEngine
    from analysis import MarketAnalyzer
    from stats_engine import StatisticsCalculator
    from charts import ChartGenerator
    from report import ReportGenerator
except ImportError as e:
    print(f"Import Error: {e}. Ensure all project files are in {PROJECT_ROOT}")
    sys.exit(1)

def main() -> None:
    """
    Main orchestration function for the Previous Day Pivot Research Engine.
    """
    # 1. Setup Infrastructure
    utils.setup_directories()
    logger = utils.setup_logger()
    
    logger.info("=" * 60)
    logger.info("INITIALIZING PREVIOUS DAY PIVOT RESEARCH ENGINE")
    logger.info("=" * 60)
    
    try:
        # 2. Load and Clean Data
        logger.info("Step 1: Loading Historical Data")
        raw_data = utils.load_and_clean_data(logger)
        
        # 3. Calculate Pivots
        logger.info("Step 2: Calculating Daily Pivots and Merging")
        pivot_engine = PivotEngine(raw_data, logger)
        daily_ohlc = pivot_engine.calculate_daily_ohlc()
        pivot_engine.calculate_pivots()
        merged_data = pivot_engine.merge_pivots_intraday()
        
        # 4. Intraday Analysis
        logger.info("Step 3: Running Intraday Market Analysis")
        analyzer = MarketAnalyzer(merged_data, logger)
        metrics_df = analyzer.run_analysis()
        
        if metrics_df.empty:
            logger.error("Analysis resulted in an empty DataFrame. Terminating.")
            sys.exit(1)
            
        # 5. Statistical Calculations
        logger.info("Step 4: Computing Statistical Probabilities")
        stats_calc = StatisticsCalculator(metrics_df, logger)
        stats_results = stats_calc.calculate_all()
        
        # 6. Generate Charts
        logger.info("Step 5: Generating Analytical Charts")
        chart_gen = ChartGenerator(metrics_df, stats_results, logger)
        chart_gen.generate_all_charts()
        
        # 7. Generate Reports
        logger.info("Step 6: Generating Comprehensive Reports")
        report_gen = ReportGenerator(metrics_df, stats_results, logger)
        report_gen.generate_all()
        
        logger.info("=" * 60)
        logger.info("RESEARCH ENGINE EXECUTION COMPLETED SUCCESSFULLY")
        logger.info(f"Check output directories in: {PROJECT_ROOT}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error("FATAL ERROR DURING EXECUTION")
        logger.error(str(e))
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()