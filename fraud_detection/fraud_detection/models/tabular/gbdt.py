"""Gradient-boosted tree baselines. Strong, fast, non-graph reference point
that every graph model in later phases has to beat to justify its cost -- and
the sub-model that keeps serving the legacy flat-`features`-array Flask
payload shape (Phase 8), which doesn't carry entity IDs to build a graph from.

`fit_arrays` (separate from `fit`) exists so training scripts can reuse one
preprocessor fit on a single train split across multiple models (GBDT, GAN
augmentation, GNN) for a fair, leakage-free comparison, and so the GAN
augmentation script can train on `real + synthetic` arrays without needing a
dataframe for the synthetic rows.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from ...data.preprocessing import FittedPreprocessor, fit_preprocessor
from ...schema import FraudDatasetSchema


@dataclass
class XGBoostFraudModel:
    schema: FraudDatasetSchema
    preprocessor: Optional[FittedPreprocessor] = None
    model: Optional[XGBClassifier] = None
    n_estimators: int = 300
    max_depth: int = 6
    learning_rate: float = 0.05

    def fit(self, train_df: pd.DataFrame) -> "XGBoostFraudModel":
        self.preprocessor = fit_preprocessor(train_df, self.schema)
        X = self.preprocessor.transform(train_df)
        y = train_df[self.schema.label_col].to_numpy()
        return self.fit_arrays(X, y)

    def fit_arrays(self, X: np.ndarray, y: np.ndarray) -> "XGBoostFraudModel":
        assert self.preprocessor is not None, "preprocessor must be set (via fit(), or assigned) before fit_arrays"
        n_pos = max(int(y.sum()), 1)
        n_neg = len(y) - n_pos
        scale_pos_weight = n_neg / n_pos

        self.model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            n_jobs=-1,
            random_state=42,
        )
        self.model.fit(X, y)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X = self.preprocessor.transform(df)
        return self.model.predict_proba(X)[:, 1]


@dataclass
class LightGBMFraudModel:
    schema: FraudDatasetSchema
    preprocessor: Optional[FittedPreprocessor] = None
    model: Optional[LGBMClassifier] = None
    n_estimators: int = 300

    def fit(self, train_df: pd.DataFrame) -> "LightGBMFraudModel":
        self.preprocessor = fit_preprocessor(train_df, self.schema)
        X = self.preprocessor.transform(train_df)
        y = train_df[self.schema.label_col].to_numpy()
        return self.fit_arrays(X, y)

    def fit_arrays(self, X: np.ndarray, y: np.ndarray) -> "LightGBMFraudModel":
        assert self.preprocessor is not None, "preprocessor must be set (via fit(), or assigned) before fit_arrays"
        self.model = LGBMClassifier(
            n_estimators=self.n_estimators,
            max_depth=-1,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
            verbose=-1,
        )
        self.model.fit(X, y)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X = self.preprocessor.transform(df)
        return self.model.predict_proba(X)[:, 1]
