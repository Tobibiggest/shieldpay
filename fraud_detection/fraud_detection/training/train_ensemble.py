"""Trains the calibrated stacking ensemble (XGBoost + GraphSAGE + HGT +
autoencoder + IsolationForest) and reports it alongside every individual
base model on the same held-out test split.

Two-stage stacking, to avoid leakage:
  1. "Fold" stage -- base models are fit on the first ~75% of the train
     window only, then scored on the remaining ~25% ("stack holdout"), rows
     none of them trained on. The meta-learner is fit on those out-of-sample
     scores, so it never sees a base model being overconfident about its own
     training data.
  2. "Final" stage -- base models are refit on the FULL train window (better
     generalization for the models actually shipped); the already-fit
     meta-learner scores their outputs on the validation split, and an
     isotonic calibrator is fit on that. Final numbers are reported on the
     untouched test split.

    python -m fraud_detection.training.train_ensemble --data data/synthetic_relational.csv
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import torch

from ..data.adapters import get_adapter
from ..data.graph.build_graph import build_hetero_graph
from ..data.graph.homogeneous import build_transaction_projection_graph
from ..data.preprocessing import fit_preprocessor, temporal_split_indices
from ..evaluation.evaluate import FraudEvaluationHarness
from ..evaluation.report import write_report
from ..models.anomaly.autoencoder import TabularAutoencoder
from ..models.anomaly.isolation_forest import IsolationForestAnomalyDetector
from ..models.ensemble.calibration import IsotonicCalibrator
from ..models.ensemble.fraud_ensemble import BASE_MODEL_ORDER, FraudEnsembleModel
from ..models.ensemble.stacking import StackingMetaLearner
from ..models.gnn.sage import GraphSAGEFraudModel
from ..models.hgnn.hgt import HGTFraudModel
from ..models.tabular.gbdt import XGBoostFraudModel
from ..utils.seed import set_global_seed
from .common import train_node_classifier_hetero, train_node_classifier_homogeneous


def _fit_base_models(
    df, schema, preprocessor, homo_data, hetero_data, train_mask, val_mask, device, epochs, lr, verbose
):
    """Fits xgboost/graphsage/hgt/autoencoder/isolation_forest using only rows
    where `train_mask` is True (val_mask is still used for GNN early
    stopping, since it always lies chronologically after every possible
    `train_mask` this function is called with)."""
    train_positions = np.nonzero(train_mask.numpy())[0]
    X_train = preprocessor.transform(df.iloc[train_positions])
    y_train = df[schema.label_col].to_numpy()[train_positions]

    xgb = XGBoostFraudModel(schema=schema, preprocessor=preprocessor)
    xgb.fit_arrays(X_train, y_train)

    x, edge_index, y_homo = homo_data.x.to(device), homo_data.edge_index.to(device), homo_data.y.to(device)
    sage = GraphSAGEFraudModel(in_channels=x.shape[1]).to(device)
    train_node_classifier_homogeneous(
        sage, x, edge_index, y_homo, train_mask, epochs, lr, val_mask=val_mask, patience=20, verbose=verbose
    )

    metadata = hetero_data.metadata()
    num_nodes_dict = {node_type: hetero_data[node_type].num_nodes for node_type in metadata[0]}
    hgt = HGTFraudModel(metadata, hetero_data["transaction"].x.shape[1], num_nodes_dict).to(device)
    train_node_classifier_hetero(
        hgt, hetero_data, hetero_data["transaction"].y, train_mask, epochs, lr,
        val_mask=val_mask, patience=20, verbose=verbose,
    )

    autoencoder = TabularAutoencoder(input_dim=X_train.shape[1])
    autoencoder.fit(X_train[y_train == 0], verbose=verbose)

    isolation_forest = IsolationForestAnomalyDetector()
    isolation_forest.fit(X_train)

    return xgb, sage, hgt, autoencoder, isolation_forest


def _score_base_models(xgb, sage, hgt, autoencoder, isolation_forest, df, preprocessor, homo_data, hetero_data, positions):
    rows = df.iloc[positions]
    X = preprocessor.transform(rows)
    xgb_scores = xgb.predict_proba(rows)
    sage_scores = sage.predict_proba(homo_data.x, homo_data.edge_index).numpy()[positions]
    hgt_scores = hgt.predict_proba(hetero_data).numpy()[positions]
    ae_scores = autoencoder.anomaly_score(X)
    if_scores = isolation_forest.anomaly_score(X)
    return np.stack([xgb_scores, sage_scores, hgt_scores, ae_scores, if_scores], axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--dataset-name", default="synthetic_relational")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--report-dir", default="artifacts/ensemble")
    parser.add_argument("--model-dir", default="artifacts/ensemble/latest")
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

    split_point = int(len(train_idx) * 0.75)
    stack_train_idx, stack_holdout_idx = train_idx[:split_point], train_idx[split_point:]

    def _mask(idx: np.ndarray) -> torch.Tensor:
        m = torch.zeros(n, dtype=torch.bool)
        m[idx] = True
        return m

    full_train_mask, val_mask = _mask(train_idx), _mask(val_idx)
    fold_train_mask = _mask(stack_train_idx)

    preprocessor = fit_preprocessor(df.iloc[train_idx], schema)
    homo_data, _ = build_transaction_projection_graph(df, schema, preprocessor=preprocessor)
    hetero_data, _, _ = build_hetero_graph(df, schema, preprocessor=preprocessor)
    homo_data, hetero_data = homo_data.to(device), hetero_data.to(device)

    y_test = df[schema.label_col].to_numpy()[test_idx]
    harness = FraudEvaluationHarness()
    results = {}

    print("Stage 1: fitting fold-phase base models (for meta-learner training)...")
    fold_models = _fit_base_models(
        df, schema, preprocessor, homo_data, hetero_data, fold_train_mask, val_mask, device,
        args.epochs, args.lr, verbose=False,
    )
    meta_X_holdout = _score_base_models(*fold_models, df, preprocessor, homo_data, hetero_data, stack_holdout_idx)
    meta_y_holdout = df[schema.label_col].to_numpy()[stack_holdout_idx]

    meta_learner = StackingMetaLearner(base_model_names=list(BASE_MODEL_ORDER))
    meta_learner.fit(meta_X_holdout, meta_y_holdout)

    print("Stage 2: fitting final-phase base models (for deployment)...")
    final_models = _fit_base_models(
        df, schema, preprocessor, homo_data, hetero_data, full_train_mask, val_mask, device,
        args.epochs, args.lr, verbose=True,
    )
    xgb, sage, hgt, autoencoder, isolation_forest = final_models

    test_meta_X = _score_base_models(*final_models, df, preprocessor, homo_data, hetero_data, test_idx)
    for name, scores in zip(BASE_MODEL_ORDER, test_meta_X.T):
        results[name] = harness.evaluate(y_test, scores, model_name=name)

    val_meta_X = _score_base_models(*final_models, df, preprocessor, homo_data, hetero_data, val_idx)
    raw_val_scores = meta_learner.predict_proba(val_meta_X)
    y_val = df[schema.label_col].to_numpy()[val_idx]
    calibrator = IsotonicCalibrator().fit(raw_val_scores, y_val)

    raw_test_scores = meta_learner.predict_proba(test_meta_X)
    calibrated_test_scores = calibrator.calibrate(raw_test_scores)
    results["ensemble_uncalibrated"] = harness.evaluate(y_test, raw_test_scores, model_name="ensemble_uncalibrated")
    results["ensemble_calibrated"] = harness.evaluate(y_test, calibrated_test_scores, model_name="ensemble_calibrated")

    ensemble = FraudEnsembleModel(
        xgb_model=xgb,
        graphsage_model=sage,
        hgt_model=hgt,
        autoencoder=autoencoder,
        isolation_forest=isolation_forest,
        meta_learner=meta_learner,
        calibrator=calibrator,
    )
    sanity_scores = ensemble.predict_proba(df, homo_data, hetero_data, test_idx)
    assert np.allclose(sanity_scores, calibrated_test_scores, atol=1e-6), "FraudEnsembleModel wiring mismatch"

    print()
    print(harness.compare(results))
    for name, result in results.items():
        write_report(result, args.report_dir, run_name=name)

    # Every component here (sklearn/XGBoost objects, torch.nn.Module models,
    # PyG Data/HeteroData graphs, the dataframe snapshot) is joblib-picklable
    # as plain Python objects, so the whole serving bundle is one dump/load
    # rather than a bespoke per-component serialization scheme.
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = model_dir / "ensemble_bundle.joblib"
    joblib.dump(
        {"ensemble": ensemble, "df": df, "homo_data": homo_data, "hetero_data": hetero_data, "schema": schema},
        bundle_path,
    )
    print(f"Saved ensemble bundle -> {bundle_path}")


if __name__ == "__main__":
    main()
