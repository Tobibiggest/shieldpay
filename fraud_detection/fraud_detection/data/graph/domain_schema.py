"""Generalized, domain-agnostic graph schema -- additive to, not a
replacement for, `FraudDatasetSchema` (`schema.py`). `FraudDatasetSchema` has
exactly 5 fixed entity-id slots (sender/recipient/device/ip/merchant), which
is all the payments domain needs but doesn't generalize to a domain with a
genuinely different entity shape (e.g. AML's originator/beneficiary, or a
future insurance domain's claimant/provider/repair-shop). `DomainGraphSchema`
replaces the fixed slots with an arbitrary list of `EntityRole`s, so a new
domain is a new schema instance, not new graph-building code.

`schema.py` and `build_graph.py` are intentionally left untouched by this
module -- see `from_fraud_dataset_schema` below, which expresses the existing
payments schema in this generic shape instead of migrating it, so the
existing hardcoded pipeline and all its tests keep working unmodified.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

from ...schema import FraudDatasetSchema


@dataclass
class EntityRole:
    """One entity type an event (e.g. a transaction) references.

    Multiple roles can share a `node_type` (e.g. payments' "sender" and
    "recipient" roles both merge into one "account" node type) -- their
    id_col values get concatenated into one unified node index.

    Edge names default to a sensible auto-generated pair if left unset; the
    payments converter below sets them explicitly to match `build_hetero_graph`'s
    existing literals (`sends`/`rev_sends`, `uses_device`/`rev_uses_device`, ...)
    so the two builders produce identical edge-type tuples for the equivalence test.
    Avoid "__" inside role/edge names -- PyG's HGTConv joins edge-type triples
    with "__" internally and could collide with a role/edge name containing it.
    """

    role_name: str
    node_type: str
    id_col: str
    entity_to_event_edge: Optional[str] = None  # source=entity, target=event; default: role_name
    event_to_entity_edge: Optional[str] = None  # source=event, target=entity; default: f"rev_{role_name}"

    def resolved_entity_to_event_edge(self) -> str:
        return self.entity_to_event_edge or self.role_name

    def resolved_event_to_entity_edge(self) -> str:
        return self.event_to_entity_edge or f"rev_{self.role_name}"


@dataclass
class CollusionEdgeSpec:
    """Derived entity-entity edges: connect pairs of `anchor_role_name`
    entities that co-occur on events sharing the same `shared_role_name`
    value (generalizes payments' `shares_device`/`shares_ip` account-account
    edges, which are the mechanism that exposes planted fraud rings to the
    heterogeneous GNN).

    `max_gap`: None reproduces payments' current untimed behavior (any two
    entities that *ever* shared the value are connected); a domain like AML,
    where "shared a beneficiary across 6 months" is meaningless but "shared
    a beneficiary within 48 hours" is a real smurfing signal, sets an actual
    timedelta. `max_group_size` is per-spec (not a shared global constant)
    since a legitimate smurfing fan-in group is far wider than payments'
    device/IP sharing scale.
    """

    anchor_role_name: str
    shared_role_name: str
    edge_name: str
    max_gap: Optional[pd.Timedelta] = None
    max_group_size: int = 25


@dataclass
class DomainGraphSchema:
    domain_name: str
    event_node_type: str
    label_col: str
    numeric_feature_cols: List[str]
    categorical_feature_cols: List[str] = field(default_factory=list)
    event_id_col: Optional[str] = None
    timestamp_col: Optional[str] = None
    entity_roles: List[EntityRole] = field(default_factory=list)
    collusion_edges: List[CollusionEdgeSpec] = field(default_factory=list)

    @property
    def feature_cols(self) -> List[str]:
        return list(self.numeric_feature_cols) + list(self.categorical_feature_cols)

    def has_relational_fields(self) -> bool:
        return bool(self.entity_roles)


def from_fraud_dataset_schema(schema: FraudDatasetSchema) -> DomainGraphSchema:
    """Expresses the existing payments `FraudDatasetSchema` in the generic
    shape, with edge names set explicitly to match `build_hetero_graph`'s
    literals -- this is what makes exact equivalence checking possible
    between the old hardcoded builder and the new generic one.
    """
    entity_roles: List[EntityRole] = []
    collusion_edges: List[CollusionEdgeSpec] = []

    # Field semantics: entity_to_event_edge = (entity -> event) edge name,
    # event_to_entity_edge = (event -> entity) edge name. build_hetero_graph's
    # naming is asymmetric per role (sender's "forward" edge goes entity->event;
    # recipient/device/ip/merchant's "forward" edge goes event->entity) --
    # preserved exactly here so the equivalence test can assert identical tuples.
    if schema.sender_id_col:
        entity_roles.append(
            EntityRole(
                role_name="sender",
                node_type="account",
                id_col="sender_id",
                entity_to_event_edge="sends",
                event_to_entity_edge="rev_sends",
            )
        )
    if schema.recipient_id_col:
        entity_roles.append(
            EntityRole(
                role_name="recipient",
                node_type="account",
                id_col="recipient_id",
                event_to_entity_edge="sent_to",
                entity_to_event_edge="rev_sent_to",
            )
        )
    if schema.device_id_col:
        entity_roles.append(
            EntityRole(
                role_name="device",
                node_type="device",
                id_col="device_id",
                event_to_entity_edge="uses_device",
                entity_to_event_edge="rev_uses_device",
            )
        )
    if schema.ip_col:
        entity_roles.append(
            EntityRole(
                role_name="ip",
                node_type="ip",
                id_col="ip_address",
                event_to_entity_edge="uses_ip",
                entity_to_event_edge="rev_uses_ip",
            )
        )
    if schema.merchant_id_col:
        entity_roles.append(
            EntityRole(
                role_name="merchant",
                node_type="merchant",
                id_col="merchant_id",
                event_to_entity_edge="at_merchant",
                entity_to_event_edge="rev_at_merchant",
            )
        )

    if schema.sender_id_col and schema.device_id_col:
        collusion_edges.append(
            CollusionEdgeSpec(anchor_role_name="sender", shared_role_name="device", edge_name="shares_device")
        )
    if schema.sender_id_col and schema.ip_col:
        collusion_edges.append(
            CollusionEdgeSpec(anchor_role_name="sender", shared_role_name="ip", edge_name="shares_ip")
        )

    return DomainGraphSchema(
        domain_name="payments",
        event_node_type="transaction",
        label_col=schema.label_col,
        numeric_feature_cols=list(schema.numeric_feature_cols),
        categorical_feature_cols=list(schema.categorical_feature_cols),
        event_id_col=schema.transaction_id_col,
        timestamp_col=schema.timestamp_col,
        entity_roles=entity_roles,
        collusion_edges=collusion_edges,
    )
