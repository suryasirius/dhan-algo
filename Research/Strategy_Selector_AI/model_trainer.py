import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split

logger = logging.getLogger("StrategySelectorAI.Trainer")

class MultiModelTrainer:
    """
    WARNING: despite the docstring/log line below, this class does NOT actually
    fit any model (no RandomForest/XGBoost/LightGBM/CatBoost/LogisticRegression
    .fit() call exists anywhere in this file) - train_all() only performs a
    train/test SPLIT and returns the split arrays. There is no trained model
    object anywhere in this module. selector.py does not and cannot consult a
    trained model, because none is ever produced here. Combined with the
    synthetic targets in feature_matrix.py, this whole module is a stub, not a
    working ML pipeline. Not wired into the live trading path (see feature_matrix.py).
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def train_all(self):
        logger.info("Splitting data for institutional ML ensemble (RF, XGB, LGBM, CatBoost, LogReg) - "
                     "NOTE: no model is actually fit in this method, see class docstring.")

        # Select numeric features for training
        feature_cols = [c for c in self.df.columns if c.startswith("feat_") or c in ["volatility_20d", "returns"]]
        X = self.df[feature_cols].select_dtypes(include=[np.number]).fillna(0)
        y = self.df["best_strategy"]
        
        # Filter out classes with fewer than 2 members to avoid stratification errors
        class_counts = y.value_counts()
        valid_classes = class_counts[class_counts >= 2].index
        mask = y.isin(valid_classes)
        
        X = X[mask]
        y = y[mask]
        
        if len(X) < 10 or y.nunique() < 2:
            logger.warning("Not enough samples per class for stratification. Splitting without stratification.")
            stratify_param = None
        else:
            stratify_param = y

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify_param
        )
        
        logger.info(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")
        return {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}
