"""Trains HGT (primary heterogeneous GNN / "HNN") and RGCN (cheaper fallback)
on the full transaction/account/device/ip/merchant graph, and reports them
alongside XGBoost and the homogeneous GraphSAGE/GAT baselines -- one
comparison table across every model family built so far.

    python -m fraud_detection.training.train_hgnn --data data/synthetic_relational.csv
"""

import argparse

import numpy as np
import torch

from ..data.adapters import get_adapter
from ..data.graph.build_graph import build_hetero_graph
from ..data.graph.homogeneous import build_transaction_projection_graph
from ..data.preprocessing import fit_preprocessor, temporal_split_indices
from ..evaluation.evaluate import FraudEvaluationHarness
from ..evaluation.report import write_report
from ..models.gnn.gat import GATFraudModel
from ..models.gnn.sage import GraphSAGEFraudModel
from ..models.hgnn.hgt import HGTFraudModel
from ..models.hgnn.rgcn import RGCNFraudModel
from ..models.tabular.gbdt import XGBoostFraudModel
from ..utils.seed import set_global_seed
from .common import train_node_classifier_hetero, train_node_classifier_homogeneous


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--dataset-name", default="synthetic_relational")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--report-dir", default="artifacts/hgnn")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_global_seed(args.seed)
    device = torch.device("cpu")

    adapter = get_adapter(args.dataset_name)
    df = adapter.load_canonical(args.data)
    schema = adapter.get_canonical_schema()
    if schema.timestamp_col and schema.timestamp_col in df.columns:
        df = df.sort_values(schema.timestamp_col).reset_index(drop=True)

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
    results = {}

    xgb = XGBoostFraudModel(schema=schema, preprocessor=preprocessor)
    xgb.fit_arrays(
        preprocessor.transform(df.iloc[train_idx]), df[schema.label_col].to_numpy()[train_idx]
    )
    results["xgboost"] = harness.evaluate(y_test, xgb.predict_proba(df.iloc[test_idx]), model_name="xgboost")

    homo_data, _ = build_transaction_projection_graph(df, schema, preprocessor=preprocessor)
    x, edge_index, y_homo = homo_data.x.to(device), homo_data.edge_index.to(device), homo_data.y.to(device)

    print("Training GraphSAGE...")
    sage = GraphSAGEFraudModel(in_channels=x.shape[1]).to(device)
    train_node_classifier_homogeneous(
        sage, x, edge_index, y_homo, train_mask, args.epochs, args.lr, val_mask=val_mask, patience=20
    )
    results["graphsage"] = harness.evaluate(
        y_test, sage.predict_proba(x, edge_index).cpu().numpy()[test_mask_np], model_name="graphsage"
    )

    print("Training GAT...")
    gat = GATFraudModel(in_channels=x.shape[1]).to(device)
    train_node_classifier_homogeneous(
        gat, x, edge_index, y_homo, train_mask, args.epochs, args.lr, val_mask=val_mask, patience=20
    )
    results["gat"] = harness.evaluate(
        y_test, gat.predict_proba(x, edge_index).cpu().numpy()[test_mask_np], model_name="gat"
    )

    hetero_data, _, _ = build_hetero_graph(df, schema, preprocessor=preprocessor)
    hetero_data = hetero_data.to(device)
    metadata = hetero_data.metadata()
    num_nodes_dict = {node_type: hetero_data[node_type].num_nodes for node_type in metadata[0]}
    transaction_in_channels = hetero_data["transaction"].x.shape[1]
    y_hetero = hetero_data["transaction"].y

    print("Training HGT...")
    hgt = HGTFraudModel(metadata, transaction_in_channels, num_nodes_dict).to(device)
    train_node_classifier_hetero(
        hgt, hetero_data, y_hetero, train_mask, args.epochs, args.lr, val_mask=val_mask, patience=20
    )
    results["hgt"] = harness.evaluate(
        y_test, hgt.predict_proba(hetero_data).cpu().numpy()[test_mask_np], model_name="hgt"
    )

    print("Training RGCN...")
    rgcn = RGCNFraudModel(metadata, transaction_in_channels, num_nodes_dict).to(device)
    train_node_classifier_hetero(
        rgcn, hetero_data, y_hetero, train_mask, args.epochs, args.lr, val_mask=val_mask, patience=20
    )
    results["rgcn"] = harness.evaluate(
        y_test, rgcn.predict_proba(hetero_data).cpu().numpy()[test_mask_np], model_name="rgcn"
    )

    print()
    print(harness.compare(results))
    for name, result in results.items():
        write_report(result, args.report_dir, run_name=name)


if __name__ == "__main__":
    main()
