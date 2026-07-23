"""Tests the save/load round trip that Flask serving depends on: train tiny
versions of every base model, bundle them via joblib (the same mechanism
`training/train_ensemble.py` uses), and confirm `FraudEnsemblePredictor.load`
can reconstruct a working predictor from disk.
"""

import joblib
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fraud_detection.data.adapters import SyntheticRelationalAdapter  # noqa: E402
from fraud_detection.data.generators.relational_synthetic_generator import (  # noqa: E402
    generate_relational_fraud_dataset,
)
from fraud_detection.data.graph.build_graph import build_hetero_graph  # noqa: E402
from fraud_detection.data.graph.homogeneous import build_transaction_projection_graph  # noqa: E402
from fraud_detection.data.preprocessing import fit_preprocessor, temporal_split_indices  # noqa: E402
from fraud_detection.models.anomaly.autoencoder import TabularAutoencoder  # noqa: E402
from fraud_detection.models.anomaly.isolation_forest import IsolationForestAnomalyDetector  # noqa: E402
from fraud_detection.models.ensemble.calibration import IsotonicCalibrator  # noqa: E402
from fraud_detection.models.ensemble.fraud_ensemble import BASE_MODEL_ORDER, FraudEnsembleModel  # noqa: E402
from fraud_detection.models.ensemble.stacking import StackingMetaLearner  # noqa: E402
from fraud_detection.models.gnn.sage import GraphSAGEFraudModel  # noqa: E402
from fraud_detection.models.hgnn.hgt import HGTFraudModel  # noqa: E402
from fraud_detection.models.tabular.gbdt import XGBoostFraudModel  # noqa: E402
from fraud_detection.serving.ensemble_predictor import FraudEnsemblePredictor  # noqa: E402
from fraud_detection.training.common import (  # noqa: E402
    train_node_classifier_hetero,
    train_node_classifier_homogeneous,
)


def _fit_tiny_bundle(tmp_path):
    df = generate_relational_fraud_dataset(
        n_transactions=500, n_accounts=70, n_devices=35, n_ips=35, n_merchants=12, n_fraud_rings=3, seed=5
    )
    adapter = SyntheticRelationalAdapter()
    schema = adapter.get_canonical_schema()
    df = adapter.to_canonical(df)

    train_idx, val_idx, test_idx = temporal_split_indices(df, schema, test_frac=0.2, val_frac=0.1)
    n = len(df)
    train_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[train_idx] = True

    preprocessor = fit_preprocessor(df.iloc[train_idx], schema)
    homo_data, _ = build_transaction_projection_graph(df, schema, preprocessor=preprocessor)
    hetero_data, _, _ = build_hetero_graph(df, schema, preprocessor=preprocessor)

    X_train = preprocessor.transform(df.iloc[train_idx])
    y_train = df[schema.label_col].to_numpy()[train_idx]

    xgb = XGBoostFraudModel(schema=schema, preprocessor=preprocessor, n_estimators=15)
    xgb.fit_arrays(X_train, y_train)

    sage = GraphSAGEFraudModel(in_channels=homo_data.x.shape[1], hidden_channels=8)
    train_node_classifier_homogeneous(
        sage, homo_data.x, homo_data.edge_index, homo_data.y, train_mask, epochs=5, lr=0.01, verbose=False
    )

    metadata = hetero_data.metadata()
    num_nodes_dict = {nt: hetero_data[nt].num_nodes for nt in metadata[0]}
    hgt = HGTFraudModel(
        metadata, hetero_data["transaction"].x.shape[1], num_nodes_dict, hidden_channels=8, num_heads=2
    )
    train_node_classifier_hetero(
        hgt, hetero_data, hetero_data["transaction"].y, train_mask, epochs=5, lr=0.01, verbose=False
    )

    autoencoder = TabularAutoencoder(input_dim=X_train.shape[1], epochs=5, batch_size=16)
    autoencoder.fit(X_train[y_train == 0])

    isolation_forest = IsolationForestAnomalyDetector(n_estimators=15)
    isolation_forest.fit(X_train)

    meta_learner = StackingMetaLearner(base_model_names=list(BASE_MODEL_ORDER))
    ensemble = FraudEnsembleModel(
        xgb_model=xgb, graphsage_model=sage, hgt_model=hgt, autoencoder=autoencoder,
        isolation_forest=isolation_forest, meta_learner=meta_learner,
    )
    meta_X_train = ensemble.base_scores(df, homo_data, hetero_data, train_idx)
    meta_learner.fit(meta_X_train, y_train)

    val_scores = ensemble.predict_proba(df, homo_data, hetero_data, val_idx)
    y_val = df[schema.label_col].to_numpy()[val_idx]
    ensemble.calibrator = IsotonicCalibrator().fit(val_scores, y_val)

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    joblib.dump(
        {"ensemble": ensemble, "df": df, "homo_data": homo_data, "hetero_data": hetero_data, "schema": schema},
        model_dir / "ensemble_bundle.joblib",
    )
    return model_dir, df


def test_ensemble_predictor_save_load_round_trip(tmp_path):
    model_dir, df = _fit_tiny_bundle(tmp_path)

    predictor = FraudEnsemblePredictor.load(model_dir)

    known_ids = predictor.sample_known_transaction_ids(limit=5)
    assert len(known_ids) == 5
    for txn_id in known_ids:
        proba = predictor.predict_proba_by_transaction_id(txn_id)
        assert proba is not None
        assert 0.0 <= proba <= 1.0

    assert predictor.predict_proba_by_transaction_id("not_a_real_transaction_id") is None
