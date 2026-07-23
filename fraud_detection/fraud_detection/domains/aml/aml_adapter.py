"""Adapter for the AML synthetic transaction dataset -- the proof that a
domain can plug into the generalized graph platform
(`data/graph/domain_schema.py`) with entity roles that don't fit payments'
fixed 5-slot schema (no merchant concept here; role NAMES are AML-flavored
-- "originator"/"beneficiary" -- even though the underlying columns are
still `sender_id`/`recipient_id`, showing role naming is decoupled from
column naming).

`BaseDatasetAdapter.to_canonical()`/`get_canonical_schema()` (`data/adapters/
base.py`) are concrete methods hardcoded against `FraudDatasetSchema.
CANONICAL_NAMES` -- calling them on a `DomainGraphSchema` would silently
no-op (`getattr` returns None for every attr) rather than crash, which is a
landmine, not a working code path. `AMLAdapter` overrides both explicitly as
documented identity functions instead, since `aml_generator.py` already
emits canonical-shaped column names directly.
"""

from pathlib import Path
from typing import Union

import pandas as pd

from ...data.adapters.base import BaseDatasetAdapter
from ...data.adapters.registry import register_adapter
from ...data.graph.domain_schema import CollusionEdgeSpec, DomainGraphSchema, EntityRole
from .aml_generator import ALL_FEATURE_COLS, CATEGORICAL_FEATURE_COLS, NUMERIC_FEATURE_COLS


class AMLAdapter(BaseDatasetAdapter):
    """Adapter for CSVs produced by `aml_generator.generate_aml_transaction_dataset`."""

    def load(self, path: Union[str, Path]) -> pd.DataFrame:
        return pd.read_csv(path, parse_dates=["timestamp"])

    def get_schema(self) -> DomainGraphSchema:
        return DomainGraphSchema(
            domain_name="aml",
            event_node_type="transaction",  # deliberately not "transfer": HGTFraudModel/RGCNFraudModel
            # hardcode the literal "transaction" as their feature-carrying node type (a real, documented
            # gap found during platform design) -- matching it avoids touching those two model files.
            label_col="label",
            numeric_feature_cols=list(NUMERIC_FEATURE_COLS),
            categorical_feature_cols=list(CATEGORICAL_FEATURE_COLS),
            event_id_col="transaction_id",
            timestamp_col="timestamp",
            entity_roles=[
                EntityRole(role_name="originator", node_type="account", id_col="sender_id"),
                EntityRole(role_name="beneficiary", node_type="account", id_col="recipient_id"),
                EntityRole(role_name="device", node_type="device", id_col="device_id"),
                EntityRole(role_name="ip", node_type="ip", id_col="ip_address"),
            ],
            collusion_edges=[
                CollusionEdgeSpec(
                    anchor_role_name="originator",
                    shared_role_name="device",
                    edge_name="shares_device",
                    max_gap=pd.Timedelta(hours=72),
                    max_group_size=50,
                ),
                CollusionEdgeSpec(
                    anchor_role_name="originator",
                    shared_role_name="ip",
                    edge_name="shares_ip",
                    max_gap=pd.Timedelta(hours=72),
                    max_group_size=50,
                ),
                # AML-specific: many originators paying the SAME beneficiary within a
                # week is exactly the smurfing signal -- payments has no equivalent
                # (its collusion edges are only device/IP based).
                CollusionEdgeSpec(
                    anchor_role_name="originator",
                    shared_role_name="beneficiary",
                    edge_name="shares_beneficiary",
                    max_gap=pd.Timedelta(days=7),
                    max_group_size=50,
                ),
            ],
        )

    def to_canonical(self, df: pd.DataFrame) -> pd.DataFrame:
        return df  # already canonical -- see module docstring

    def get_canonical_schema(self) -> DomainGraphSchema:
        return self.get_schema()  # already canonical -- see module docstring


register_adapter("aml", AMLAdapter)
