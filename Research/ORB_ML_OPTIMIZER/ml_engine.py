import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import config

class MachineLearningEngine:
    def __init__(self, trades_df: pd.DataFrame, logger: logging.Logger):
        self.trades_df = trades_df
        self.logger = logger
        self.models: Dict[str, Any] = {}

    def build_feature_matrix(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Constructs feature matrix for machine learning classification."""
        self.logger.info("Constructing feature matrix for machine learning models...")
        
        if self.trades_df.empty:
            return pd.DataFrame(), pd.Series()
            
        # METHODOLOGY FIX: sort chronologically before slicing train/test so the
        # split below is a real walk-forward split, not a shuffled one that lets
        # future trades leak into training.
        ordered = self.trades_df.sort_values('date').reset_index(drop=True)

        features = ordered[['orb_range', 'gap_pct', 'above_vwap', 'above_pivot', 'day_of_week', 'atr', 'volume']].copy()
        features.fillna(0.0, inplace=True)

        target = ordered['win']
        return features, target

    def train_models(self) -> Dict[str, float]:
        """Trains Random Forest and Logistic Regression classifiers to predict trade success probability."""
        features, target = self.build_feature_matrix()

        if len(features) < 20:
            self.logger.warning("Insufficient trades for reliable machine learning training.")
            return {}

        # METHODOLOGY FIX: chronological (walk-forward) split instead of a
        # shuffled random split - trades are time-ordered, so training on data
        # that includes days AFTER the test set is temporal leakage.
        split_idx = int(len(features) * (1.0 - config.TEST_SPLIT))
        split_idx = max(1, min(split_idx, len(features) - 1))
        X_train, X_test = features.iloc[:split_idx], features.iloc[split_idx:]
        y_train, y_test = target.iloc[:split_idx], target.iloc[split_idx:]
        
        # Random Forest Classifier
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        rf_preds = rf.predict(X_test)
        rf_acc = accuracy_score(y_test, rf_preds)
        
        # Logistic Regression Classifier
        lr = LogisticRegression(max_iter=500)
        lr.fit(X_train, y_train)
        lr_preds = lr.predict(X_test)
        lr_acc = accuracy_score(y_test, lr_preds)
        
        self.models['RandomForest'] = rf
        self.models['LogisticRegression'] = lr
        
        self.logger.info(f"Machine learning training completed. RF Accuracy: {rf_acc:.2%}, LR Accuracy: {lr_acc:.2%}")
        return {
            'RandomForest_Accuracy': float(rf_acc),
            'LogisticRegression_Accuracy': float(lr_acc)
        }