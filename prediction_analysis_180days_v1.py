"""Evaluate 15-minute candle direction predictions at every completed minute."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

INPUT_FILE = Path(__file__).with_name("training_dataset_180days_v2.csv")
TEST_FRACTION = 0.20
RANDOM_STATE = 42
ACCURACY_FILE = Path(__file__).with_name("prediction_accuracy_180days_v1.csv")
RESULTS_FILE = Path(__file__).with_name("prediction_results_180days_v1.csv")
IMPORTANCE_FILE = Path(__file__).with_name("feature_importance_180days_v1.csv")
REPORT_FILE = Path(__file__).with_name("prediction_report_180days_v1.txt")
ACCURACY_CHART_FILE = Path(__file__).with_name("accuracy_vs_minute_180days_v1.png")
CONFUSION_CHART_FILE = Path(__file__).with_name("confusion_matrix_best_180days_v1.png")


def prepare_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    prepared = frame.copy()
    atr = prepared["atr14"].replace(0, np.nan)
    prepared["body_atr"] = prepared["body"] / atr
    prepared["range_atr"] = prepared["range"] / atr
    prepared["upper_wick_atr"] = prepared["upper_wick"] / atr
    prepared["lower_wick_atr"] = prepared["lower_wick"] / atr
    prepared["ema_spread_atr"] = prepared["ema_spread"] / atr
    prepared["close_ema9_atr"] = (prepared["close"] - prepared["ema9"]) / atr
    prepared["close_ema21_atr"] = (prepared["close"] - prepared["ema21"]) / atr
    prepared["htf_body_atr"] = prepared["htf_body_so_far"] / atr
    prepared["htf_range_atr"] = prepared["htf_range_so_far"] / atr
    prepared["htf_upper_wick_atr"] = prepared["htf_upper_wick_so_far"] / atr
    prepared["htf_lower_wick_atr"] = prepared["htf_lower_wick_so_far"] / atr
    prepared["close_from_htf_open_atr"] = (prepared["close"] - prepared["htf_open"]) / atr

    feature_columns = [
        "body_atr",
        "range_atr",
        "upper_wick_atr",
        "lower_wick_atr",
        "body_pct_of_range",
        "ema_spread_atr",
        "close_ema9_atr",
        "close_ema21_atr",
        "rsi14",
        "return_1m",
        "return_5m",
        "htf_body_atr",
        "htf_range_atr",
        "htf_upper_wick_atr",
        "htf_lower_wick_atr",
        "close_from_htf_open_atr",
    ]

    if "volume_ratio_20" in prepared.columns:
        feature_columns.append("volume_ratio_20")

    return prepared.replace([np.inf, -np.inf], np.nan), feature_columns


def build_models() -> dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2_000, random_state=RANDOM_STATE)),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=8,
                        min_samples_leaf=8,
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def calculate_metrics(actual: pd.Series, predicted: np.ndarray, probability: np.ndarray | None) -> dict[str, float]:
    metrics = {
        "accuracy": accuracy_score(actual, predicted),
        "precision": precision_score(actual, predicted, zero_division=0),
        "recall": recall_score(actual, predicted, zero_division=0),
        "f1": f1_score(actual, predicted, zero_division=0),
    }
    if probability is not None and actual.nunique() == 2:
        metrics["roc_auc"] = roc_auc_score(actual, probability)
    else:
        metrics["roc_auc"] = np.nan
    return metrics


def save_accuracy_chart(metrics: pd.DataFrame) -> None:
    plt.figure(figsize=(11, 6))
    for model_name, subset in metrics.groupby("model"):
        plt.plot(subset["minute_in_15m"], subset["accuracy"] * 100, marker="o", label=model_name)
    plt.axhline(50, color="gray", linestyle="--", linewidth=1, label="50% reference")
    plt.xticks(range(1, 15))
    plt.xlabel("Completed minute of the 15-minute candle")
    plt.ylabel("Out-of-sample accuracy (%)")
    plt.title("Final 15-minute candle direction accuracy")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(ACCURACY_CHART_FILE, dpi=160)
    plt.close()


def save_confusion_chart(results: pd.DataFrame, best_row: pd.Series) -> None:
    selected = results.loc[
        (results["minute_in_15m"] == best_row["minute_in_15m"])
        & (results["model"] == best_row["model"])
    ]
    display = ConfusionMatrixDisplay.from_predictions(
        selected["actual"],
        selected["predicted"],
        display_labels=["Bearish", "Bullish"],
        cmap="Blues",
        colorbar=False,
    )
    display.ax_.set_title(
        f"{best_row['model']} at minute {int(best_row['minute_in_15m'])}\n"
        f"Out-of-sample accuracy: {best_row['accuracy'] * 100:.2f}%"
    )
    display.figure_.tight_layout()
    display.figure_.savefig(CONFUSION_CHART_FILE, dpi=160)
    plt.close(display.figure_)


def main() -> None:
    if not INPUT_FILE.exists():
        raise SystemExit(f"Input file was not found: {INPUT_FILE.name}")

    frame = pd.read_csv(INPUT_FILE, parse_dates=["datetime", "htf_start"])
    required_columns = {"htf_start", "minute_in_15m", "target_bullish", "htf_body_so_far"}
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise SystemExit(f"Input file is missing columns: {', '.join(missing_columns)}")

    frame, feature_columns = prepare_features(frame)
    bars = np.sort(frame["htf_start"].unique())
    split_index = int(len(bars) * (1 - TEST_FRACTION))
    train_bars = set(bars[:split_index])
    test_bars = set(bars[split_index:])
    if not train_bars or not test_bars:
        raise SystemExit("The dataset is too small for a chronological train/test split.")

    metric_rows: list[dict[str, object]] = []
    result_frames: list[pd.DataFrame] = []
    importance_rows: list[dict[str, object]] = []

    for minute in range(1, 15):
        minute_frame = frame.loc[frame["minute_in_15m"] == minute].copy()
        train = minute_frame.loc[minute_frame["htf_start"].isin(train_bars)]
        test = minute_frame.loc[minute_frame["htf_start"].isin(test_bars)]
        features_train = train[feature_columns]
        features_test = test[feature_columns]
        target_train = train["target_bullish"]
        target_test = test["target_bullish"]

        baseline_prediction = (test["htf_body_so_far"] > 0).astype(int).to_numpy()
        baseline_metrics = calculate_metrics(target_test, baseline_prediction, None)
        metric_rows.append(
            {
                "model": "Current HTF Direction",
                "minute_in_15m": minute,
                "train_bars": len(train),
                "test_bars": len(test),
                "test_bullish_rate": target_test.mean(),
                **baseline_metrics,
            }
        )
        result_frames.append(
            pd.DataFrame(
                {
                    "htf_start": test["htf_start"].to_numpy(),
                    "minute_in_15m": minute,
                    "model": "Current HTF Direction",
                    "actual": target_test.to_numpy(),
                    "predicted": baseline_prediction,
                    "bullish_probability": np.nan,
                }
            )
        )

        for model_name, model in build_models().items():
            model.fit(features_train, target_train)
            prediction = model.predict(features_test)
            probability = model.predict_proba(features_test)[:, 1]
            metrics = calculate_metrics(target_test, prediction, probability)
            metric_rows.append(
                {
                    "model": model_name,
                    "minute_in_15m": minute,
                    "train_bars": len(train),
                    "test_bars": len(test),
                    "test_bullish_rate": target_test.mean(),
                    **metrics,
                }
            )
            result_frames.append(
                pd.DataFrame(
                    {
                        "htf_start": test["htf_start"].to_numpy(),
                        "minute_in_15m": minute,
                        "model": model_name,
                        "actual": target_test.to_numpy(),
                        "predicted": prediction,
                        "bullish_probability": probability,
                    }
                )
            )

            if model_name == "Random Forest":
                forest = model.named_steps["model"]
                for feature, importance in zip(feature_columns, forest.feature_importances_, strict=True):
                    importance_rows.append(
                        {
                            "minute_in_15m": minute,
                            "feature": feature,
                            "importance": importance,
                        }
                    )

        print(f"Completed minute {minute}/14")

    metrics_frame = pd.DataFrame(metric_rows).sort_values(["minute_in_15m", "model"])
    results_frame = pd.concat(result_frames, ignore_index=True)
    importance_frame = pd.DataFrame(importance_rows)
    importance_summary = (
        importance_frame.groupby("feature", as_index=False)["importance"].mean().sort_values("importance", ascending=False)
    )
    importance_frame = pd.concat(
        [importance_frame.assign(scope="by_minute"), importance_summary.assign(minute_in_15m="all_minutes", scope="average")],
        ignore_index=True,
    )

    metrics_frame.to_csv(ACCURACY_FILE, index=False)
    results_frame.to_csv(RESULTS_FILE, index=False)
    importance_frame.to_csv(IMPORTANCE_FILE, index=False)
    save_accuracy_chart(metrics_frame)

    model_metrics = metrics_frame.loc[metrics_frame["model"] != "Current HTF Direction"]
    best_row = model_metrics.sort_values(["accuracy", "f1"], ascending=False).iloc[0]
    save_confusion_chart(results_frame, best_row)

    average_accuracy = (
        metrics_frame.groupby("model", as_index=False)["accuracy"].mean().sort_values("accuracy", ascending=False)
    )
    best_by_minute = (
        metrics_frame.sort_values(["minute_in_15m", "accuracy", "f1"], ascending=[True, False, False])
        .groupby("minute_in_15m", as_index=False)
        .first()
    )

    report_lines = [
        "15-Minute Candle Direction Prediction Analysis",
        "",
        f"Input: {INPUT_FILE.name}",
        f"Total 15-minute bars: {len(bars):,}",
        f"Chronological training bars: {len(train_bars):,}",
        f"Chronological test bars: {len(test_bars):,}",
        f"Test period: {min(test_bars)} to {max(test_bars)}",
        "",
        "Average out-of-sample accuracy by model:",
    ]
    report_lines.extend(
        f"- {row.model}: {row.accuracy * 100:.2f}%" for row in average_accuracy.itertuples(index=False)
    )
    report_lines.extend(
        [
            "",
            "Best non-baseline result:",
            f"- {best_row['model']} at minute {int(best_row['minute_in_15m'])}: {best_row['accuracy'] * 100:.2f}% accuracy, {best_row['roc_auc']:.3f} ROC-AUC",
            "",
            "Best model by minute:",
        ]
    )
    report_lines.extend(
        f"- Minute {int(row.minute_in_15m)}: {row.model} at {row.accuracy * 100:.2f}%" for row in best_by_minute.itertuples(index=False)
    )
    report_lines.extend(
        [
            "",
            "Top average Random Forest features:",
        ]
    )
    report_lines.extend(
        f"- {row.feature}: {row.importance:.4f}" for row in importance_summary.head(10).itertuples(index=False)
    )
    report_lines.extend(
        [
            "",
            "Interpretation:",
            "- Results are out-of-sample because the newest 20% of 15-minute bars are never used for training.",
            "- Minute 15 is intentionally excluded because its final direction is already known.",
            "- Accuracy alone is not a trading edge; costs, slippage, confidence thresholds, and repeated walk-forward tests must be evaluated before live use.",
        ]
    )
    REPORT_FILE.write_text("\n".join(report_lines), encoding="utf-8")

    print("\nAnalysis complete")
    print(f"Best result: {best_row['model']} at minute {int(best_row['minute_in_15m'])} = {best_row['accuracy'] * 100:.2f}%")
    print(f"Saved: {ACCURACY_FILE.name}")
    print(f"Saved: {REPORT_FILE.name}")
    print(f"Saved: {ACCURACY_CHART_FILE.name}")


if __name__ == "__main__":
    main()
