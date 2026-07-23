"""Dataset-agnostic column contract.

Every adapter, graph builder, and model in this package reads dataframe columns
only through a `FraudDatasetSchema` instance -- never a hardcoded column name.
That's what lets the same graph-construction and model code run unmodified
against the bundled synthetic dataset, IEEE-CIS, or any future dataset: a new
dataset only needs a new adapter that produces one of these.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class FraudDatasetSchema:
    label_col: str
    numeric_feature_cols: List[str]
    categorical_feature_cols: List[str] = field(default_factory=list)

    transaction_id_col: Optional[str] = None
    sender_id_col: Optional[str] = None
    recipient_id_col: Optional[str] = None
    device_id_col: Optional[str] = None
    ip_col: Optional[str] = None
    merchant_id_col: Optional[str] = None
    timestamp_col: Optional[str] = None

    @property
    def feature_cols(self) -> List[str]:
        return list(self.numeric_feature_cols) + list(self.categorical_feature_cols)

    @property
    def entity_id_cols(self) -> Dict[str, Optional[str]]:
        """Maps graph node-type name -> source column, for columns present in this dataset."""
        return {
            "account": self.sender_id_col or self.recipient_id_col,
            "device": self.device_id_col,
            "ip": self.ip_col,
            "merchant": self.merchant_id_col,
        }

    def has_relational_fields(self) -> bool:
        return any(
            [
                self.sender_id_col,
                self.recipient_id_col,
                self.device_id_col,
                self.ip_col,
                self.merchant_id_col,
            ]
        )

    # Canonical column names used internally once an adapter's `to_canonical`
    # has run -- graph builders and training code target these, not the
    # dataset-specific originals.
    CANONICAL_NAMES = {
        "transaction_id_col": "transaction_id",
        "sender_id_col": "sender_id",
        "recipient_id_col": "recipient_id",
        "device_id_col": "device_id",
        "ip_col": "ip_address",
        "merchant_id_col": "merchant_id",
        "timestamp_col": "timestamp",
        "label_col": "label",
    }
