"""Adapter for the IEEE-CIS Fraud Detection dataset
(Kaggle: https://www.kaggle.com/c/ieee-fraud-detection). Real transaction-
level fraud data (~3.5% fraud) with an identity table whose `card1`/
`DeviceInfo`/`addr1`/`P_emaildomain` fields give entity-like proxies --
chosen over PaySim (also fully synthetic, adds little beyond this project's
own generator) and Elliptic (node-level AML labels covering only ~23% of
nodes, not transaction-level).

Unlike this project's own synthetic generator, IEEE-CIS was not designed
with explicit sender/recipient/device IDs, so these mappings are best-effort
proxies, not ground truth:

  card1         -> sender_id      the payment card used; the closest thing
                                   to a stable "who is transacting" ID
  DeviceInfo    -> device_id      raw device string from the identity table
  addr1         -> recipient_id   billing region, a coarse proxy for "who/
                                   where the money is going"
  P_emaildomain -> merchant_id    purchaser's email domain, a weak proxy for
                                   merchant/platform context
  TransactionDT -> timestamp      seconds since a reference point, not a
                                   real datetime -- fine for `temporal_split`,
                                   which only needs relative ordering
  isFraud       -> label

This repo does not fetch the CSVs -- see docs/DATASETS.md for how to obtain
them with your own Kaggle credentials. Place `train_transaction.csv` (and
optionally `train_identity.csv`) under `fraud_detection/data/ieee_cis/`, or
pass an explicit directory to `load()`.

Feature columns are deliberately a small, documented subset of the ~400 raw
columns (IEEE-CIS's anonymized `V*` columns are out of scope here) -- extend
`NUMERIC_FEATURE_COLS`/`CATEGORICAL_FEATURE_COLS` if you need more. IEEE-CIS
is heavily sparse, and `FittedPreprocessor` does not itself handle NaNs, so
missing values in the chosen columns are filled (0 for numeric, "missing"
for categorical) here.
"""

from pathlib import Path
from typing import Union

import pandas as pd

from ...schema import FraudDatasetSchema
from .base import BaseDatasetAdapter

NUMERIC_FEATURE_COLS = ["TransactionAmt", "C1", "C2", "C13", "C14", "D1", "D2", "D4", "D15"]
CATEGORICAL_FEATURE_COLS = ["ProductCD", "card4", "card6", "M1", "M2", "M3", "M4", "M6"]


class IEEECISAdapter(BaseDatasetAdapter):
    def load(
        self,
        path: Union[str, Path] = "data/ieee_cis",
        transaction_file: str = "train_transaction.csv",
        identity_file: str = "train_identity.csv",
    ) -> pd.DataFrame:
        base = Path(path)
        df = pd.read_csv(base / transaction_file)

        identity_path = base / identity_file
        if identity_path.exists():
            identity = pd.read_csv(identity_path)
            df = df.merge(identity, on="TransactionID", how="left")

        for col in NUMERIC_FEATURE_COLS:
            if col not in df.columns:
                df[col] = 0.0
        for col in CATEGORICAL_FEATURE_COLS:
            if col not in df.columns:
                df[col] = "missing"
        df[NUMERIC_FEATURE_COLS] = df[NUMERIC_FEATURE_COLS].fillna(0.0)
        df[CATEGORICAL_FEATURE_COLS] = df[CATEGORICAL_FEATURE_COLS].fillna("missing")

        if "DeviceInfo" not in df.columns:
            df["DeviceInfo"] = "unknown_device"
        df["DeviceInfo"] = df["DeviceInfo"].fillna("unknown_device")
        if "addr1" in df.columns:
            df["addr1"] = df["addr1"].fillna(-1)
        if "P_emaildomain" in df.columns:
            df["P_emaildomain"] = df["P_emaildomain"].fillna("unknown_domain")

        return df

    def get_schema(self) -> FraudDatasetSchema:
        return FraudDatasetSchema(
            label_col="isFraud",
            numeric_feature_cols=list(NUMERIC_FEATURE_COLS),
            categorical_feature_cols=list(CATEGORICAL_FEATURE_COLS),
            transaction_id_col="TransactionID",
            sender_id_col="card1",
            recipient_id_col="addr1",
            device_id_col="DeviceInfo",
            merchant_id_col="P_emaildomain",
            timestamp_col="TransactionDT",
        )
