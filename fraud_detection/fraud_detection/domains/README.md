# Adding a new domain

This package generalizes graph-based fraud detection beyond payments. The
model layer (`models/gnn/`, `models/hgnn/`, `models/tabular/`,
`models/anomaly/`, `models/ensemble/`) is already domain-agnostic -- it
consumes a PyG `HeteroData`/feature matrix and knows nothing about
"transaction" semantics. What's domain-specific is the **schema and data
layer**. Adding a new domain (insurance claims, marketplace fraud,
account-takeover, ...) means writing that layer once; you do not touch
`schema.py`, `build_graph.py`, or any existing model/training code.

`domains/aml/` is the worked example this recipe is extracted from -- read
its four files alongside this doc if anything below is ambiguous.

## The recipe

**1. Decide your entity roles and event type.**

Every domain models one kind of *event* row (a transaction, a claim, a
login) that references some number of *entities* (accounts, devices,
providers, products). Write these down before touching code:

| AML (built) | Insurance claims (example) | Marketplace (example) |
|---|---|---|
| event = transaction | event = claim | event = listing/order |
| originator (account) | claimant | buyer |
| beneficiary (account) | provider | seller |
| device | repair_shop | product |
| ip | -- | -- |

**2. Write a data generator** (or a real-data adapter if you have actual
data instead of needing synthetic data).

If synthetic: follow `domains/aml/aml_generator.py`'s **camouflage
discipline**, non-negotiably. Planted fraud rows must be feature-camouflaged
-- roughly 85% of them should be statistically indistinguishable from
legitimate rows on their own per-row features, detectable only via graph
structure (shared entities, timing, chains). This is not a style
preference: the first version of the payments generator skipped this and
XGBoost hit AUPRC=1.0 on features alone, leaving nothing for the graph
models to add value on top of -- the exact same mistake reappeared in AML's
first draft (a features-only XGBoost hit AUPRC=0.96) and was caught by an
automated test, not manual review. **Write that test for your domain too**
(see `tests/test_aml_generator.py::test_camouflage_discipline_...`): fit
`XGBoostFraudModel` on raw features only and assert AUPRC stays low on your
planted fraud. Don't rely on remembering to check this by hand.

**3. Write an adapter** (`data/adapters/base.py::BaseDatasetAdapter`
subclass) whose `get_schema()` returns a `DomainGraphSchema`
(`data/graph/domain_schema.py`), not a `FraudDatasetSchema` -- the latter is
payments' fixed 5-slot schema and won't fit a genuinely different entity
shape. Concretely:

```python
DomainGraphSchema(
    domain_name="your_domain",
    event_node_type="transaction",   # see the gotcha below -- don't rename this without also
                                       # touching models/hgnn/hgt.py and rgcn.py
    label_col="label",
    numeric_feature_cols=[...],
    categorical_feature_cols=[...],
    timestamp_col="timestamp",
    entity_roles=[
        EntityRole(role_name="claimant", node_type="party", id_col="claimant_id"),
        EntityRole(role_name="provider", node_type="party", id_col="provider_id"),  # merges into
                                                                                        # the same node_type
        EntityRole(role_name="repair_shop", node_type="repair_shop", id_col="repair_shop_id"),
    ],
    collusion_edges=[
        CollusionEdgeSpec(
            anchor_role_name="claimant", shared_role_name="repair_shop",
            edge_name="shares_repair_shop",
            max_gap=pd.Timedelta(days=30),  # None = untimed, connects entities that EVER
                                              # co-occurred -- almost always too broad; give it a real window.
            max_group_size=50,
        ),
    ],
)
```

`BaseDatasetAdapter.to_canonical()`/`get_canonical_schema()` are concrete
methods hardcoded against `FraudDatasetSchema.CANONICAL_NAMES` -- calling
them on a `DomainGraphSchema` silently no-ops rather than crashing (a
landmine, not a working path). **Override both explicitly** in your adapter
(see `aml_adapter.py`); the simplest fix, if your generator already emits
canonical-shaped column names, is to make both the identity function.
Register your adapter with `register_adapter("your_domain", YourAdapter)`
(`data/adapters/registry.py`) called from your own adapter module -- never
edit `registry.py` itself.

**Gotcha:** `HGTFraudModel`/`RGCNFraudModel` (`models/hgnn/hgt.py`,
`models/hgnn/rgcn.py`) hardcode the literal string `"transaction"` as the
feature-carrying node type. Name your `event_node_type` `"transaction"` too
unless you're willing to patch those two files (add an `event_node_type`
constructor parameter, default `"transaction"` for backward compatibility).
`GraphSAGEFraudModel`/`GATFraudModel` have no such constraint.

**4. (Optional) Write domain-specific pattern detectors.**

If your domain has a structural fraud pattern that's naturally a
graph-traversal/sequence problem rather than a per-row statistic (AML's
layering chains and rapid-pass-through are examples), write it against the
**dataframe**, not the built `HeteroData` -- see `domains/aml/aml_patterns.py`'s
module docstring for why (`HeteroData` is a tensor container with no
adjacency API; reconstructing one from it is more work than building it from
the dataframe directly). Match the finding-dict shape used everywhere else
in this codebase: `{pattern_type, description, severity, affected_row_indices}`.

**5. Reuse the existing training scripts, don't write new model code.**

`training/train_aml.py` is the template: load your adapter, temporal-split,
fit `XGBoostFraudModel` as a baseline, build a homogeneous graph via
`data/graph/homogeneous.py::build_transaction_projection_graph` (works
unchanged if your canonical columns are named `sender_id`/`device_id`/
`ip_address` like AML's are -- reuse those names if your domain has an
equivalent concept, even if the `EntityRole.role_name` is domain-flavored),
fit `GraphSAGEFraudModel`, build a heterogeneous graph via
`data/graph/build_domain_graph.py::build_domain_hetero_graph`, fit
`HGTFraudModel`. `training/common.py::train_node_classifier_hetero`/
`train_node_classifier_homogeneous` (shared, validation-based early
stopping) work for any domain unmodified.

**6. Validate, don't assume.**

Run the comparison table. If the graph models don't meaningfully beat the
tabular baseline, that's a signal your synthetic data isn't structurally
realistic yet (see step 2), not that the platform doesn't work -- AML's
result (`training/train_aml.py`, full dataset): XGBoost AUPRC 0.13,
GraphSAGE 0.25, **HGT 0.61** -- confirms the same story payments told
(camouflaged, purely-structural fraud is only catchable via the graph), and
is the evidence this platform generalizes rather than just claims to.

## What's deliberately out of scope per domain (for now)

The full anomaly-detector/stacking-ensemble/calibration treatment
(`models/anomaly/`, `models/ensemble/`, `training/train_ensemble.py` --
payments' Phase 6) is not required to validate a new domain. Proving the
graph adds value over a tabular baseline (steps 1-6 above) is the bar; the
full ensemble is a reasonable fast-follow once a domain is proven, not a
prerequisite.
