import sys
import config
from utils import setup_logger
from loader import DataLoader
from feature_matrix import FeatureMatrixEngine
from model_trainer import MultiModelTrainer
from selector import StrategySelectorEngine
from charts import ChartGenerator
from report import InstitutionalReporter

def main():
    logger = setup_logger()
    logger.info("=== INITIALIZING STRATEGY SELECTOR AI PLATFORM ===")
    print("\n--- STRATEGY SELECTOR AI PIPELINE TRACE ---")

    loader = DataLoader()
    raw_df = loader.load_inputs()

    feat_engine = FeatureMatrixEngine(raw_df)
    features_df = feat_engine.build_features()

    trainer = MultiModelTrainer(features_df)
    models = trainer.train_all()

    selector = StrategySelectorEngine(features_df)
    recommendations_df = selector.generate_recommendations()

    chart_gen = ChartGenerator(features_df, recommendations_df)
    chart_gen.generate_all_charts()

    reporter = InstitutionalReporter(features_df, recommendations_df)
    reporter.export_all_reports()

    logger.info("=== STRATEGY SELECTOR AI PIPELINE COMPLETED SUCCESSFULLY ===")
    print("=== STRATEGY SELECTOR AI PIPELINE COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()
