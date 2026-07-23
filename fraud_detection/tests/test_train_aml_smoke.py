"""Smoke test for train_aml.py's building blocks on a tiny AML dataset --
fast enough for the regular suite; the real comparison-table run (larger
dataset, full epoch budget) is done manually/in CI as a separate step, the
same way payments' train_hgnn.py comparison run was.
"""

import numpy as np
import pytest
import torch

from fraud_detection.data.graph.build_domain_graph import build_domain_hetero_graph
from fraud_detection.data.graph.homogeneous import build_transaction_projection_graph
from fraud_detection.data.preprocessing import fit_preprocessor, temporal_split_indices
from fraud_detection.domains.aml.aml_adapter import AMLAdapter
from fraud_detection.domains.aml.aml_generator import generate_aml_transaction_dataset
from fraud_detection.evaluation.evaluate import FraudEvaluationHarness
from fraud_detection.models.gnn.sage import GraphSAGEFraudModel
from fraud_detection.models.hgnn.hgt import HGTFraudModel
from fraud_detection.models.tabular.gbdt import XGBoostFraudModel
from fraud_detection.training.common import train_node_classifier_hetero, train_node_classifier_homogeneous


def test_train_aml_pipeline_smoke(tmp_path):
    df = generate_aml_transaction_dataset(
        n_transactions=800, n_accounts=100, n_devices=50, n_ips=50,
        n_layering_chains=4, n_smurf_rings=2, seed=9,
    )
    csv_path = tmp_path / "aml.csv"
    df.to_csv(csv_path, index=False)

    adapter = AMLAdapter()
    df = adapter.load_canonical(csv_path)
    schema = adapter.get_canonical_schema()

    train_idx, val_idx, test_idx = temporal_split_indices(df, schema)
    n = len(df)
    train_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask = torch.zeros(n, dtype=torch.bool)
    val_mask[val_idx] = True
    test_mask_np = np.zeros(n, dtype=bool)
    test_mask_np[test_idx] = True

    preprocessor = fit_preprocessor(df.iloc[train_idx], schema)
    y_test = df[schema.label_col].to_numpy()[test_idx]
    harness = FraudEvaluationHarness()

    xgb = XGBoostFraudModel(schema=schema, preprocessor=preprocessor, n_estimators=20)
    xgb.fit_arrays(preprocessor.transform(df.iloc[train_idx]), df[schema.label_col].to_numpy()[train_idx])
    xgb_result = harness.evaluate(y_test, xgb.predict_proba(df.iloc[test_idx]), model_name="xgboost")
    assert 0.0 <= xgb_result["auprc"] <= 1.0 or np.isnan(xgb_result["auprc"])

    homo_data, _ = build_transaction_projection_graph(df, schema, preprocessor=preprocessor)
    sage = GraphSAGEFraudModel(in_channels=homo_data.x.shape[1], hidden_channels=8)
    train_node_classifier_homogeneous(
        sage, homo_data.x, homo_data.edge_index, homo_data.y, train_mask, epochs=5, lr=0.01,
        val_mask=val_mask, patience=20, verbose=False,
    )
    sage_scores = sage.predict_proba(homo_data.x, homo_data.edge_index).numpy()
    assert sage_scores.shape == (n,)

    hetero_data, _, _ = build_domain_hetero_graph(df, schema, preprocessor=preprocessor)
    metadata = hetero_data.metadata()
    num_nodes_dict = {nt: hetero_data[nt].num_nodes for nt in metadata[0]}
    hgt = HGTFraudModel(metadata, hetero_data["transaction"].x.shape[1], num_nodes_dict, hidden_channels=8, num_heads=2)
    train_node_classifier_hetero(
        hgt, hetero_data, hetero_data["transaction"].y, train_mask, epochs=5, lr=0.01,
        val_mask=val_mask, patience=20, verbose=False,
    )
    hgt_scores = hgt.predict_proba(hetero_data).numpy()
    assert hgt_scores.shape == (n,)
