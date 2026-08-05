import time
import pandas as pd
import config
from utils import setup_directories, setup_logger, load_and_clean_data
from orb_engine import ORBEngine
from optimizer import StrategyOptimizer
from ml_engine import MachineLearningEngine
from monte_carlo import MonteCarloSimulator
from charts import ChartGenerator
from report import ReportGenerator

def main():
    start_time = time.time()
    
    # 1. Setup Directories & Logger
    setup_directories()
    logger = setup_logger()
    
    logger.info("============================================================")
    logger.info("INITIALIZING ORB QUANTITATIVE MACHINE LEARNING OPTIMIZER")
    logger.info("============================================================")
    
    try:
        # 2. Load Historical Data
        logger.info("Step 1: Loading Historical Data")
        df = load_and_clean_data(logger)
        
        # 3. Initialize ORB Simulation Engine & Features
        logger.info("Step 2: Preparing Market Features & Running Simulations")
        orb_engine = ORBEngine(df, logger)
        processed_data = orb_engine.prepare_market_features()
        
        # 4. Run Grid Search Optimization
        logger.info("Step 3: Running Multi-Dimensional Grid Search")
        optimizer = StrategyOptimizer(orb_engine, logger)
        grid_results = optimizer.run_grid_search()

        # METHODOLOGY FIX: grid_results used to be computed and then discarded -
        # the ML/Monte Carlo steps below ran on a hardcoded config regardless of
        # what the grid search found. We now actually select the best combination
        # (ranked by SQN, a robustness-adjusted metric that penalizes low trade
        # counts and high variance more than raw net profit would; profit_factor
        # breaks ties) and use it for everything downstream.
        if grid_results is not None and not grid_results.empty:
            ranked = grid_results.sort_values(
                by=["sqn", "profit_factor"], ascending=[False, False]
            )
            best = ranked.iloc[0]
            logger.info(
                "Step 3b: Best grid combo selected by SQN -> "
                f"orb={int(best['orb_minutes'])}m, sl={best['stop_loss']}, "
                f"trail={best['trail_stop']}, target={best['target']}, "
                f"time_filter={best['time_filter']} "
                f"(SQN={best['sqn']:.3f}, PF={best['profit_factor']:.2f}, "
                f"trades={int(best['trades'])})"
            )
            sample_trades = orb_engine.simulate_orb_strategy(
                orb_minutes=int(best["orb_minutes"]),
                stop_loss_pts=best["stop_loss"],
                trail_stop_pts=best["trail_stop"],
                target_type=best["target"],
                time_filter=best["time_filter"]
            )
        else:
            logger.warning(
                "Grid search returned no valid combinations (every combo had <5 trades). "
                "Falling back to the default ORB config for reporting."
            )
            sample_trades = orb_engine.simulate_orb_strategy(
                orb_minutes=15,
                stop_loss_pts=10,
                trail_stop_pts=5,
                target_type="2R",
                time_filter="09:30"
            )
        trades_df = pd.DataFrame(sample_trades) if sample_trades else pd.DataFrame()
        
        # 5. Machine Learning Training
        logger.info("Step 4: Training Machine Learning Classifiers")
        ml_engine = MachineLearningEngine(trades_df, logger)
        ml_metrics = ml_engine.train_models()
        
        # 6. Monte Carlo Simulations
        logger.info("Step 5: Executing Monte Carlo Risk Simulations")
        mc_simulator = MonteCarloSimulator(trades_df, logger)
        mc_results = mc_simulator.run_simulations()
        
        # 7. Generate Charts and Visualizations
        logger.info("Step 6: Generating Institutional Charts and PDF")
        chart_gen = ChartGenerator(trades_df, mc_results, logger)
        chart_gen.generate_all_charts()
        
        # 8. Compile Final Reports
        logger.info("Step 7: Compiling Final Research Reports")
        report_gen = ReportGenerator(trades_df, mc_results, logger)
        report_gen.generate_all()
        
        elapsed_time = time.time() - start_time
        logger.info("============================================================")
        logger.info(f"OPTIMIZATION & RESEARCH COMPLETED SUCCESSFULLY IN {elapsed_time:.2f} SECONDS")
        logger.info(f"Check output and reports in: {config.PROJECT_ROOT}")
        logger.info("============================================================")
        
    except Exception as e:
        logger.error(f"Critical execution failure: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()