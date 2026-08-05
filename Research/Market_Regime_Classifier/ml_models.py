import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
logger = logging.getLogger("MarketRegime.ML")
class RegimeMLTrainer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.results = {}
        self.feature_names = ["Trend_Strength", "Trend_Score", "Volatility_Score", "Momentum_Score", "Gap_Score", "Volume_Score", "Risk_Score"]
    def train_and_evaluate(self):
        # NOTE (methodology fix): this used to build a KFold(shuffle=True)
        # splitter and only ever consume `next(kf.split(X))` - i.e. a single
        # shuffled 80/20 split mislabeled as "5-fold CV". Shuffling also
        # breaks chronological order for what is time-series data (rows are
        # market sessions in date order), letting the model train on rows
        # that occur AFTER its test rows. We now use TimeSeriesSplit, which
        # never shuffles and always trains on the past / tests on the future,
        # and we actually loop over every fold and average the metrics.
        logger.info("Training ML models using TimeSeriesSplit (chronological) cross-validation...")
        X = self.df[self.feature_names].fillna(0)
        y = self.df["Market_Type"]
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        all_classes = np.unique(y_encoded)

        n_splits = 5
        tscv = TimeSeriesSplit(n_splits=n_splits)

        classifier_names = ["Random Forest", "XGBoost", "LightGBM", "Logistic Regression", "Decision Tree"]
        fold_metrics = {name: {"Accuracy": [], "Precision": [], "Recall": [], "F1 Score": []} for name in classifier_names}
        confusion_totals = {name: None for name in classifier_names}
        folds_used = {name: 0 for name in classifier_names}

        for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

            # If a class is missing from this training fold, do NOT patch it
            # in by pulling a row from the full/test-inclusive X/y_encoded -
            # that would leak a test-set row into training. Instead just log
            # a warning; the classifier will simply never predict that class
            # in this fold, which sklearn handles fine.
            train_classes = np.unique(y_train)
            missing_classes = [c for c in all_classes if c not in train_classes]
            if missing_classes:
                logger.warning(
                    f"Fold {fold_idx}: class(es) {le.inverse_transform(missing_classes).tolist()} "
                    f"absent from this training fold. Skipping injection (would leak test data); "
                    f"these classes simply cannot be predicted by models in this fold."
                )

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            classifiers = {
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
                "LightGBM": LGBMClassifier(random_state=42, verbose=-1),
                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                "Decision Tree": DecisionTreeClassifier(random_state=42)
            }
            for name, clf in classifiers.items():
                if name == "Logistic Regression":
                    clf.fit(X_train_scaled, y_train)
                    y_pred = clf.predict(X_test_scaled)
                else:
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)

                fold_metrics[name]["Accuracy"].append(accuracy_score(y_test, y_pred))
                fold_metrics[name]["Precision"].append(precision_score(y_test, y_pred, average="weighted", zero_division=0))
                fold_metrics[name]["Recall"].append(recall_score(y_test, y_pred, average="weighted", zero_division=0))
                fold_metrics[name]["F1 Score"].append(f1_score(y_test, y_pred, average="weighted", zero_division=0))
                cm = confusion_matrix(y_test, y_pred, labels=all_classes)
                confusion_totals[name] = cm if confusion_totals[name] is None else (confusion_totals[name] + cm)
                folds_used[name] += 1

        for name in classifier_names:
            self.results[name] = {
                "Accuracy": float(np.mean(fold_metrics[name]["Accuracy"])),
                "Precision": float(np.mean(fold_metrics[name]["Precision"])),
                "Recall": float(np.mean(fold_metrics[name]["Recall"])),
                "F1 Score": float(np.mean(fold_metrics[name]["F1 Score"])),
                "Confusion Matrix": confusion_totals[name],
                "Folds Evaluated": folds_used[name],
            }
            logger.info(
                f"{name}: avg over {folds_used[name]} chronological folds - "
                f"Accuracy={self.results[name]['Accuracy']:.4f}, F1={self.results[name]['F1 Score']:.4f}"
            )
        return self.results

