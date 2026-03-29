"""Baseline methods used for comparison against the full investigation agent."""

from __future__ import annotations

from abc import ABC, abstractmethod
import os

import numpy as np
import pandas as pd
from loguru import logger

from src.config import settings


VALID_LABELS = {"epidemic", "environmental", "sampling", "mixed", "uncertain"}


def normalize_label(value: str | None) -> str:
    if value is None:
        return "uncertain"
    label = str(value).strip().lower()
    return label if label in VALID_LABELS else "uncertain"


class BaseBaseline(ABC):
    name: str = ""

    @abstractmethod
    def predict(self, events_df: pd.DataFrame, merged_db: pd.DataFrame) -> list[str]:
        ...


class RuleBasedBaseline(BaseBaseline):
    """Rule baseline using the existing weak-label logic."""

    name = "rule_based"

    def predict(self, events_df: pd.DataFrame, merged_db: pd.DataFrame) -> list[str]:
        from src.labeling.auto_label import auto_label_event

        predictions: list[str] = []
        for _, event in events_df.iterrows():
            result = auto_label_event(event, merged_db)
            predictions.append(normalize_label(result.get("label")))
        return predictions


class MajorityClassBaseline(BaseBaseline):
    """Always predicts the most frequent class from training labels."""

    name = "majority_class"

    def __init__(self, majority_label: str):
        self.majority_label = normalize_label(majority_label)

    def predict(self, events_df: pd.DataFrame, merged_db: pd.DataFrame) -> list[str]:
        return [self.majority_label] * len(events_df)


class MLBaseline(BaseBaseline):
    """Traditional ML baseline with engineered temporal features."""

    name = "ml"

    def __init__(self, classifier_type: str = "random_forest", random_state: int = 42):
        self.classifier_type = classifier_type
        self.random_state = random_state
        self.name = classifier_type
        self.model = None
        self.feature_names: list[str] = []

    def _extract_features(self, event: pd.Series, merged_db: pd.DataFrame) -> dict[str, float]:
        site_id = event["site_id"]
        peak_date = pd.Timestamp(event["peak_date"])
        lag_days = settings.agent.clinical_lag_days

        site_data = merged_db[merged_db["site_id"] == site_id].sort_values("date")
        pre_window = site_data[
            (site_data["date"] >= peak_date - pd.Timedelta(days=7))
            & (site_data["date"] <= peak_date)
        ]
        post_window = site_data[
            (site_data["date"] > peak_date)
            & (site_data["date"] <= peak_date + pd.Timedelta(days=lag_days))
        ]

        features: dict[str, float] = {
            "peak_zscore": float(event.get("peak_zscore", 0) or 0),
            "duration_days": float(event.get("duration_days", 1) or 1),
            "vote_count_max": float(event.get("vote_count_max", 0) or 0),
        }

        if "precipitation_mm" in pre_window.columns:
            precip = pre_window["precipitation_mm"].dropna()
            if len(precip) > 0:
                features["precip_sum_7d"] = float(precip.sum())
                features["precip_max_7d"] = float(precip.max())
                features["has_heavy_rain"] = float(precip.max() > 25)
            else:
                features["precip_sum_7d"] = 0.0
                features["precip_max_7d"] = 0.0
                features["has_heavy_rain"] = 0.0

        if "temp_avg_c" in pre_window.columns:
            temp = pre_window["temp_avg_c"].dropna()
            features["temp_avg_7d"] = float(temp.mean()) if len(temp) > 0 else np.nan
            features["temp_max_7d"] = float(temp.max()) if len(temp) > 0 else np.nan

        if "discharge_percentile" in pre_window.columns:
            flow = pre_window["discharge_percentile"].dropna()
            features["flow_pct_max_7d"] = float(flow.max()) if len(flow) > 0 else np.nan

        target_col = "previous_day_admission_adult_covid_confirmed"
        if target_col in site_data.columns:
            pre_clin = pre_window[target_col].dropna()
            post_clin = post_window[target_col].dropna()
            features["admission_pre_mean"] = float(pre_clin.mean()) if len(pre_clin) > 0 else np.nan
            features["admission_post_mean"] = float(post_clin.mean()) if len(post_clin) > 0 else np.nan
            if len(pre_clin) > 0 and len(post_clin) > 0 and pre_clin.mean() > 0:
                features["admission_change_pct"] = float(
                    (post_clin.mean() - pre_clin.mean()) / pre_clin.mean() * 100
                )
            else:
                features["admission_change_pct"] = 0.0

        return features

    def _build_feature_frame(self, events_df: pd.DataFrame, merged_db: pd.DataFrame) -> pd.DataFrame:
        feature_dicts = [self._extract_features(event, merged_db) for _, event in events_df.iterrows()]
        return pd.DataFrame(feature_dicts).fillna(0)

    def _build_model(self):
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        if self.classifier_type == "logistic_regression":
            clf = LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
                class_weight="balanced",
            )
        elif self.classifier_type == "random_forest":
            clf = RandomForestClassifier(
                n_estimators=300,
                random_state=self.random_state,
                class_weight="balanced_subsample",
            )
        elif self.classifier_type in {"xgboost", "gradient_boosting"}:
            clf = GradientBoostingClassifier(random_state=self.random_state)
        else:
            raise ValueError(f"Unknown classifier_type: {self.classifier_type}")

        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", clf),
            ]
        )

    def fit(self, events_df: pd.DataFrame, labels: list[str], merged_db: pd.DataFrame) -> None:
        X = self._build_feature_frame(events_df, merged_db)
        y = np.array([normalize_label(x) for x in labels], dtype=object)
        self.model = self._build_model()
        self.model.fit(X, y)
        self.feature_names = list(X.columns)
        logger.info(
            f"ML baseline fitted: type={self.classifier_type}, "
            f"samples={len(X)}, features={len(self.feature_names)}"
        )

    def predict(self, events_df: pd.DataFrame, merged_db: pd.DataFrame) -> list[str]:
        if self.model is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        X = self._build_feature_frame(events_df, merged_db)
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_names]
        return [normalize_label(x) for x in self.model.predict(X).tolist()]

    def predict_oof(
        self,
        events_df: pd.DataFrame,
        labels: list[str],
        merged_db: pd.DataFrame,
        max_splits: int = 5,
    ) -> list[str]:
        """Out-of-fold prediction for fairer baseline comparison."""
        from sklearn.model_selection import StratifiedKFold

        X = self._build_feature_frame(events_df, merged_db)
        y = np.array([normalize_label(x) for x in labels], dtype=object)

        label_counts = pd.Series(y).value_counts()
        min_class_count = int(label_counts.min()) if len(label_counts) > 0 else 0
        n_splits = min(max_splits, min_class_count)

        if len(y) < 8 or n_splits < 2:
            logger.warning(
                f"OOF skipped for {self.classifier_type}: small data. "
                "Falling back to train-and-predict on the same set."
            )
            self.fit(events_df, labels, merged_db)
            return self.predict(events_df, merged_db)

        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=self.random_state,
        )
        oof_pred = np.empty(len(y), dtype=object)

        for train_idx, test_idx in splitter.split(X, y):
            fold_model = self._build_model()
            fold_model.fit(X.iloc[train_idx], y[train_idx])
            oof_pred[test_idx] = fold_model.predict(X.iloc[test_idx])

        return [normalize_label(x) for x in oof_pred.tolist()]


class _LLMBaselineMixin:
    """Shared prompt construction and inference helpers for LLM baselines."""

    def __init__(self, model: str | None = None):
        self.model = model or settings.agent.model

    def _build_client(self):
        from openai import OpenAI

        api_key = (
            settings.agent.api_key
            or os.environ.get("OPENROUTER_API_KEY")
            or os.environ.get("OPENAI_API_KEY", "")
        )
        if not api_key:
            return None
        return OpenAI(base_url=settings.agent.api_base, api_key=api_key)

    def _summarize_window(self, window: pd.DataFrame) -> str:
        if window.empty:
            return "No data available."
        numeric_cols = [
            c
            for c in [
                "pcr_conc_lin_log1p",
                "precipitation_mm",
                "temp_avg_c",
                "discharge_percentile",
                "previous_day_admission_adult_covid_confirmed",
            ]
            if c in window.columns
        ]
        if not numeric_cols:
            return "No numeric signals available."

        summary_parts: list[str] = []
        for col in numeric_cols:
            s = window[col].dropna()
            if len(s) == 0:
                continue
            summary_parts.append(
                f"{col}: mean={float(s.mean()):.3f}, max={float(s.max()):.3f}, min={float(s.min()):.3f}"
            )
        return "; ".join(summary_parts) if summary_parts else "No numeric signals available."

    def _build_event_prompt(self, event: pd.Series, merged_db: pd.DataFrame) -> str:
        site_id = event["site_id"]
        peak_date = pd.Timestamp(event["peak_date"])
        window = merged_db[
            (merged_db["site_id"] == site_id)
            & (merged_db["date"] >= peak_date - pd.Timedelta(days=7))
            & (merged_db["date"] <= peak_date + pd.Timedelta(days=14))
        ]
        return (
            f"Event: site={site_id}; peak_date={peak_date.date()}; "
            f"peak_zscore={float(event.get('peak_zscore', 0) or 0):.2f}; "
            f"duration_days={int(event.get('duration_days', 1) or 1)}.\n"
            f"Signal summary: {self._summarize_window(window)}"
        )

    def _classify_with_prompt(self, client, prompt: str) -> str:
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=32,
            )
            content = response.choices[0].message.content or ""
            return normalize_label(content)
        except Exception as exc:
            logger.error(f"LLM baseline prediction failed: {exc}")
            return "uncertain"


class ZeroShotLLMBaseline(_LLMBaselineMixin, BaseBaseline):
    """Direct one-pass LLM classifier baseline without tool usage."""

    name = "zero_shot_llm"

    def predict(self, events_df: pd.DataFrame, merged_db: pd.DataFrame) -> list[str]:
        client = self._build_client()
        if client is None:
            logger.warning("No API key found. Zero-shot LLM baseline falls back to 'uncertain'.")
            return ["uncertain"] * len(events_df)

        predictions: list[str] = []
        for _, event in events_df.iterrows():
            prompt = (
                "Classify this wastewater anomaly into one label: "
                "epidemic, environmental, sampling, mixed, uncertain.\n"
                f"{self._build_event_prompt(event, merged_db)}\n"
                "Answer with one label only."
            )
            predictions.append(self._classify_with_prompt(client, prompt))
        return predictions


class FewShotLLMBaseline(_LLMBaselineMixin, BaseBaseline):
    """Few-shot LLM classifier with examples drawn only from the training fold."""

    name = "few_shot_llm"

    def __init__(self, model: str | None = None, max_examples: int = 5, random_state: int = 42):
        super().__init__(model=model)
        self.max_examples = max_examples
        self.random_state = random_state

    def _select_examples(self, train_df: pd.DataFrame) -> pd.DataFrame:
        if train_df.empty:
            return train_df

        examples: list[pd.DataFrame] = []
        for label, group in train_df.groupby("y_true"):
            ranked = group.assign(abs_z=group["peak_zscore"].abs()).sort_values(
                ["abs_z", "duration_days", "event_id"], ascending=[False, False, True]
            )
            examples.append(ranked.head(1))

        selected = pd.concat(examples, ignore_index=True) if examples else train_df.head(0)
        if len(selected) >= self.max_examples:
            return selected.head(self.max_examples)

        remaining = train_df.loc[~train_df["event_id"].isin(selected["event_id"])].copy()
        if remaining.empty:
            return selected
        remaining = remaining.assign(abs_z=remaining["peak_zscore"].abs()).sort_values(
            ["abs_z", "duration_days", "event_id"], ascending=[False, False, True]
        )
        extra_needed = self.max_examples - len(selected)
        return pd.concat([selected, remaining.head(extra_needed)], ignore_index=True)

    def _format_examples(self, examples_df: pd.DataFrame, merged_db: pd.DataFrame) -> str:
        parts: list[str] = []
        for idx, (_, event) in enumerate(examples_df.iterrows(), start=1):
            parts.append(
                f"Example {idx}\n"
                f"{self._build_event_prompt(event, merged_db)}\n"
                f"Correct label: {event['y_true']}\n"
            )
        return "\n".join(parts)

    def predict(self, events_df: pd.DataFrame, merged_db: pd.DataFrame) -> list[str]:
        raise RuntimeError("FewShotLLMBaseline requires training examples. Use predict_oof().")

    def predict_oof(
        self,
        events_df: pd.DataFrame,
        labels: list[str],
        merged_db: pd.DataFrame,
        max_splits: int = 5,
    ) -> list[str]:
        from sklearn.model_selection import StratifiedKFold

        client = self._build_client()
        if client is None:
            logger.warning("No API key found. Few-shot LLM baseline falls back to 'uncertain'.")
            return ["uncertain"] * len(events_df)

        work_df = events_df.copy().reset_index(drop=True)
        work_df["y_true"] = [normalize_label(x) for x in labels]

        label_counts = work_df["y_true"].value_counts()
        min_class_count = int(label_counts.min()) if len(label_counts) > 0 else 0
        n_splits = min(max_splits, min_class_count)
        if len(work_df) < 8 or n_splits < 2:
            logger.warning("Few-shot OOF skipped because the label set is too small.")
            return ["uncertain"] * len(work_df)

        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=self.random_state,
        )
        oof_pred = np.empty(len(work_df), dtype=object)

        for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(work_df, work_df["y_true"]), start=1):
            train_df = work_df.iloc[train_idx].copy()
            test_df = work_df.iloc[test_idx].copy()
            examples_df = self._select_examples(train_df)
            examples_text = self._format_examples(examples_df, merged_db)
            logger.info(
                f"Few-shot fold {fold_idx}/{n_splits}: train={len(train_df)}, "
                f"test={len(test_df)}, examples={len(examples_df)}"
            )

            for row_idx, event in test_df.iterrows():
                prompt = (
                    "Classify this wastewater anomaly into one label: "
                    "epidemic, environmental, sampling, mixed, uncertain.\n\n"
                    "Use the examples below as reference cases.\n\n"
                    f"{examples_text}\n"
                    "Now classify the new event.\n"
                    f"{self._build_event_prompt(event, merged_db)}\n"
                    "Answer with one label only."
                )
                oof_pred[row_idx] = self._classify_with_prompt(client, prompt)

        return [normalize_label(x) for x in oof_pred.tolist()]
