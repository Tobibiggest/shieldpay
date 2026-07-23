"""Shared tabular preprocessing: scaling numeric columns, encoding categoricals.

Used by both the GBDT baseline (Phase 2) and as the source of `transaction`
node features for the graph builder -- one fit, two consumers, so a GBDT
feature vector and a transaction node's feature vector are always the same
representation of the same row.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from ..schema import FraudDatasetSchema


@dataclass
class FittedPreprocessor:
    scaler: MinMaxScaler
    encoder: OneHotEncoder
    numeric_cols: List[str]
    categorical_cols: List[str]
    output_columns: List[str] = field(default_factory=list)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        parts = []
        if self.numeric_cols:
            parts.append(self.scaler.transform(df[self.numeric_cols]))
        if self.categorical_cols:
            parts.append(self.encoder.transform(df[self.categorical_cols]))
        return np.concatenate(parts, axis=1).astype(np.float32)


def fit_preprocessor(df: pd.DataFrame, schema: FraudDatasetSchema) -> FittedPreprocessor:
    numeric_cols = list(schema.numeric_feature_cols)
    categorical_cols = list(schema.categorical_feature_cols)

    scaler = MinMaxScaler()
    if numeric_cols:
        scaler.fit(df[numeric_cols])

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    if categorical_cols:
        encoder.fit(df[categorical_cols])

    output_columns = list(numeric_cols)
    if categorical_cols:
        output_columns += list(encoder.get_feature_names_out(categorical_cols))

    return FittedPreprocessor(
        scaler=scaler,
        encoder=encoder,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        output_columns=output_columns,
    )


def temporal_split_indices(
    df: pd.DataFrame, schema: FraudDatasetSchema, test_frac: float = 0.2, val_frac: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Row-position (not label) indices for a chronological train/val/test
    split, assuming `df` is already sorted by `schema.timestamp_col` (or in
    its natural order if there is no timestamp column). Splitting by time
    rather than randomly matters here because a random split would let future
    transactions leak into training through shared-entity graph edges (an
    account's later fraud-ring membership would otherwise be visible while
    training on its earlier, legitimate transactions).

    Returns positional indices rather than a boolean mask so callers can use
    them directly with both `.iloc` (dataframes) and array indexing (graph
    node features), against the same underlying row order.
    """
    n = len(df)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    n_train = n - n_test - n_val

    train_idx = np.arange(0, n_train)
    val_idx = np.arange(n_train, n_train + n_val)
    test_idx = np.arange(n_train + n_val, n)
    return train_idx, val_idx, test_idx


def temporal_split(
    df: pd.DataFrame, schema: FraudDatasetSchema, test_frac: float = 0.2, val_frac: float = 0.1
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Dataframe-slice convenience wrapper around `temporal_split_indices`."""
    if schema.timestamp_col and schema.timestamp_col in df.columns:
        ordered = df.sort_values(schema.timestamp_col).reset_index(drop=True)
    else:
        ordered = df.reset_index(drop=True)

    train_idx, val_idx, test_idx = temporal_split_indices(ordered, schema, test_frac, val_frac)
    return ordered.iloc[train_idx], ordered.iloc[val_idx], ordered.iloc[test_idx]
