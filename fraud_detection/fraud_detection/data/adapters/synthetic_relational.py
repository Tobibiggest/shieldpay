from pathlib import Path
from typing import Union

import pandas as pd

from ...schema import FraudDatasetSchema
from ..generators.relational_synthetic_generator import (
    CATEGORICAL_FEATURE_COLS,
    NUMERIC_FEATURE_COLS,
)
from .base import BaseDatasetAdapter


class SyntheticRelationalAdapter(BaseDatasetAdapter):
    """Adapter for CSVs produced by `relational_synthetic_generator`."""

    def load(self, path: Union[str, Path]) -> pd.DataFrame:
        return pd.read_csv(path, parse_dates=["timestamp"])

    def get_schema(self) -> FraudDatasetSchema:
        return FraudDatasetSchema(
            label_col="label",
            numeric_feature_cols=list(NUMERIC_FEATURE_COLS),
            categorical_feature_cols=list(CATEGORICAL_FEATURE_COLS),
            transaction_id_col="transaction_id",
            sender_id_col="sender_id",
            recipient_id_col="recipient_id",
            device_id_col="device_id",
            ip_col="ip_address",
            merchant_id_col="merchant_id",
            timestamp_col="timestamp",
        )
