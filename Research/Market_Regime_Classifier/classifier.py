import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("MarketRegime.Classifier")

class RegimeAutoLabeler:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def compute_scores_and_labels(self) -> pd.DataFrame:
        print(f"Rows before feature engineering & regime labeling = {len(self.df)}")
        logger.info("Computing quantitative scores and auto-labeling market regimes...")
        
        for col in ["pct_change", "day_range", "volume", "Gap_Pct", "VWAP_Distance", "ATR"]:
            if col not in self.df.columns:
                self.df[col] = 0.0

        # NOTE (leakage fix): all normalization statistics below use EXPANDING
        # windows (stats computed only from data up to and including the
        # current row) instead of whole-dataframe .max()/.min()/.mean()/.std().
        # Using global (full-dataset) statistics leaks future information into
        # every row's features, since a row early in the dataset would be
        # normalized using min/max/mean/std values that are only knowable
        # once the *entire* dataset (including future rows) has been seen.
        # `min_periods=20` means the first ~20 rows of the dataset will have
        # NaN scores (not enough history yet); downstream code fillna(0) on
        # these, which is an acceptable, non-leaking way to handle the
        # unavoidable cold-start of a purely-causal rolling/expanding stat.
        MIN_PERIODS = 20

        pct_abs = self.df["pct_change"].abs()
        vol_mean = self.df["volume"].expanding(min_periods=MIN_PERIODS).mean()
        vol_std = self.df["volume"].expanding(min_periods=MIN_PERIODS).std()
        vol_z = (self.df["volume"] - vol_mean) / (vol_std + 1e-8)

        atr_mean = self.df["ATR"].expanding(min_periods=MIN_PERIODS).mean()
        atr_std = self.df["ATR"].expanding(min_periods=MIN_PERIODS).std()
        atr_z = (self.df["ATR"] - atr_mean) / (atr_std + 1e-8)

        pct_abs_exp_max = pct_abs.expanding(min_periods=MIN_PERIODS).max()
        atr_z_exp_min = atr_z.expanding(min_periods=MIN_PERIODS).min()
        atr_z_exp_max = atr_z.expanding(min_periods=MIN_PERIODS).max()
        gap_abs = self.df["Gap_Pct"].abs()
        gap_abs_exp_max = gap_abs.expanding(min_periods=MIN_PERIODS).max()
        vol_z_exp_min = vol_z.expanding(min_periods=MIN_PERIODS).min()
        vol_z_exp_max = vol_z.expanding(min_periods=MIN_PERIODS).max()

        self.df["Trend_Strength"] = (pct_abs / (pct_abs_exp_max + 1e-8)) * 100
        self.df["Trend_Score"] = self.df["Trend_Strength"]
        self.df["Volatility_Score"] = ((atr_z - atr_z_exp_min) / (atr_z_exp_max - atr_z_exp_min + 1e-8)) * 100
        self.df["Momentum_Score"] = (self.df["pct_change"] / (pct_abs_exp_max + 1e-8)) * 50 + 50
        self.df["Gap_Score"] = (gap_abs / (gap_abs_exp_max + 1e-8)) * 100
        self.df["Volume_Score"] = ((vol_z - vol_z_exp_min) / (vol_z_exp_max - vol_z_exp_min + 1e-8)) * 100
        self.df["Risk_Score"] = self.df["Volatility_Score"] * 0.7 + (self.df["Gap_Score"] * 0.3)
        self.df["Confidence_Score"] = (50 + (self.df["Trend_Strength"] * 0.5)).clip(50, 99.9)

        # WARNING - label/feature coupling (target leakage by construction):
        # The `Market_Type` label produced by this loop is derived directly
        # from the SAME raw quantities (pct_change, Gap_Pct, Volatility_Score,
        # day_range, Volume_Score, net_change) that ml_models.py also feeds
        # into the model as *input features* (Trend_Strength, Volatility_Score,
        # Momentum_Score, Gap_Score, Volume_Score, Risk_Score are all built
        # from these same raw columns above). This means the "ML model" in
        # ml_models.py is not predicting an independent outcome - it is
        # largely being asked to reconstruct the deterministic if/elif rule
        # below from a near-linear transform of its own inputs. Any reported
        # accuracy/F1/precision from ml_models.py must be interpreted as
        # "how well can the model recover our own labeling rule", NOT as
        # evidence of genuine predictive skill on unseen market regimes.
        # Properly decoupling this would require an independently-sourced
        # regime label (e.g. human-annotated, or derived from a different
        # information set than the model features) - that redesign is out of
        # scope for this fix.
        day_range_exp_mean = self.df["day_range"].expanding(min_periods=MIN_PERIODS).mean()
        labels, reasons = [], []
        for idx, row in self.df.iterrows():
            pct, gap, vol_score, trend_str = row["pct_change"], row["Gap_Pct"], row["Volatility_Score"], row["Trend_Strength"]
            if abs(gap) > 1.0 and pct * gap < 0:
                label, reason = "Gap Fade", f"Significant opening gap ({gap:.2f}%) reversed."
            elif abs(gap) > 1.0 and pct * gap >= 0:
                label, reason = "Gap Trend", f"Opening gap ({gap:.2f}%) sustained."
            elif pct > 1.0 and trend_str > 70:
                label, reason = "Strong Trend Up", f"Strong upward price expansion (+{pct:.2f}%)."
            elif pct < -1.0 and trend_str > 70:
                label, reason = "Strong Trend Down", f"Strong downward price collapse ({pct:.2f}%)."
            elif vol_score > 80:
                label, reason = "High Volatility", "Elevated ATR and wide intraday range."
            elif vol_score < 20:
                label, reason = "Low Volatility", "Compressed range and subdued activity."
            elif abs(pct) > 0.8 and row["Volume_Score"] > 75:
                label, reason = "Momentum Day", "High volume driving directional movement."
            elif row["day_range"] > (1.5 * day_range_exp_mean.loc[idx]) and abs(pct) < 0.3:
                label, reason = "Outside Day", "Wide daily range with flat net change."
            elif abs(pct) <= 0.3 and vol_score < 40:
                label, reason = "Inside Day", "Tight consolidation within previous session."
            elif row["net_change"] * pct < 0 and trend_str > 50:
                label, reason = "Reversal Day", "Intraday trend reversal from opening direction."
            else:
                label, reason = "Range", "Bounded oscillation without directional commitment."
            labels.append(label)
            reasons.append(reason)

        self.df["Market_Type"] = labels
        self.df["Confidence"] = self.df["Confidence_Score"] / 100.0
        self.df["Reason"] = reasons
        self.df["Top_Features"] = "Trend_Score, Volatility_Score, Volume_Score, Gap_Score"
        
        print(f"Rows after feature engineering & regime labeling = {len(self.df)}")
        return self.df

