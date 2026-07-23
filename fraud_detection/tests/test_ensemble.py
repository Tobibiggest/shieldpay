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
from fraud_detection.models.ensemble.calibration import IsotonicCalibrator, PlattCalibrator  # noqa: E402
from fraud_detection.models.ensemble.fraud_ensemble import BASE_MODEL_ORDER, FraudEnsembleModel  # noqa: E402
from fraud_detection.models.ensemble.stacking import StackingMetaLearner  # noqa: E402
from fraud_detection.models.gnn.sage import GraphSAGEFraudModel  # noqa: E402
from fraud_detection.models.hgnn.hgt import HGTFraudModel  # noqa: E402
from fraud_detection.models.tabular.gbdt import XGBoostFraudModel  # noqa: E402
from fraud_detection.training.common import (  # noqa: E402
    train_node_classifier_hetero,
    train_node_classifier_homogeneous,
)


def test_autoencoder_fit_and_anomaly_score_smoke():
    rng = np.random.default_rng(0)
    X_non_fraud = rng.random((100, 10)).astype(np.float32)
    X_test = rng.random((20, 10)).astype(np.float32)

    ae = TabularAutoencoder(input_dim=10, epochs=5, batch_size=16)
    ae.fit(X_non_fraud)

    scores = ae.anomaly_score(X_test)
    assert scores.shape == (20,)
    assert np.isfinite(scores).all()
    assert np.all(scores >= 0)  # squared reconstruction error


def test_isolation_forest_anomaly_score_smoke():
    rng = np.random.default_rng(0)
    X_train = rng.random((100, 10))
    X_test = rng.random((20, 10))

    iso = IsolationForestAnomalyDetector(n_estimators=20)
    iso.fit(X_train)

    scores = iso.anomaly_score(X_test)
    assert scores.shape == (20,)
    assert np.isfinite(scores).all()


def test_stacking_meta_learner_fit_predict():
    rng = np.random.default_rng(0)
    n = 200
    meta_X = rng.random((n, 3))
    meta_y = (meta_X[:, 0] + meta_X[:, 1] > 1.0).astype(int)

    learner = StackingMetaLearner(base_model_names=["a", "b", "c"])
    learner.fit(meta_X, meta_y)

    proba = learner.predict_proba(meta_X)
    assert proba.shape == (n,)
    assert np.all(proba >= 0) and np.all(proba <= 1)


def test_isotonic_calibrator_monotonic_and_bounded():
    rng = np.random.default_rng(0)
    raw = np.sort(rng.random(100))
    y = (raw > 0.5).astype(int)

    calibrator = IsotonicCalibrator().fit(raw, y)
    calibrated = calibrator.calibrate(raw)

    assert np.all(calibrated >= 0) and np.all(calibrated <= 1)
    # isotonic regression is monotonic non-decreasing by construction
    assert np.all(np.diff(calibrated) >= -1e-9)


def test_platt_calibrator_bounded():
    rng = np.random.default_rng(0)
    raw = rng.random(100)
    y = (raw > 0.5).astype(int)

    calibrator = PlattCalibrator().fit(raw, y)
    calibrated = calibrator.calibrate(raw)
    assert np.all(calibrated >= 0) and np.all(calibrated <= 1)


def test_fraud_ensemble_model_end_to_end_wiring():
    """Fits every base model + meta-learner + calibrator on a tiny dataset
    and checks FraudEnsembleModel assembles and scores correctly -- this is
    the integration test for the most novel wiring code in Phase 6."""
    df = generate_relational_fraud_dataset(
        n_transactions=600, n_accounts=80, n_devices=40, n_ips=40, n_merchants=15, n_fraud_rings=4, seed=3
    )
    adapter = SyntheticRelationalAdapter()
    schema = adapter.get_schema()
    df = adapter.to_canonical(df)

    train_idx, val_idx, test_idx = temporal_split_indices(df, schema, test_frac=0.2, val_frac=0.1)
    n = len(df)
    train_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask = torch.zeros(n, dtype=torch.bool)
    val_mask[val_idx] = True

    preprocessor = fit_preprocessor(df.iloc[train_idx], schema)
    homo_data, _ = build_transaction_projection_graph(df, schema, preprocessor=preprocessor)
    hetero_data, _, _ = build_hetero_graph(df, schema, preprocessor=preprocessor)

    X_train = preprocessor.transform(df.iloc[train_idx])
    y_train = df[schema.label_col].to_numpy()[train_idx]

    xgb = XGBoostFraudModel(schema=schema, preprocessor=preprocessor, n_estimators=20)
    xgb.fit_arrays(X_train, y_train)

    sage = GraphSAGEFraudModel(in_channels=homo_data.x.shape[1], hidden_channels=8)
    train_node_classifier_homogeneous(
        sage, homo_data.x, homo_data.edge_index, homo_data.y, train_mask, epochs=5, lr=0.01, verbose=False
    )

    metadata = hetero_data.metadata()
    num_nodes_dict = {nt: hetero_data[nt].num_nodes for nt in metadata[0]}
    hgt = HGTFraudModel(metadata, hetero_data["transaction"].x.shape[1], num_nodes_dict, hidden_channels=8, num_heads=2)
    train_node_classifier_hetero(hgt, hetero_data, hetero_data["transaction"].y, train_mask, epochs=5, lr=0.01, verbose=False)

    autoencoder = TabularAutoencoder(input_dim=X_train.shape[1], epochs=5, batch_size=16)
    autoencoder.fit(X_train[y_train == 0])

    isolation_forest = IsolationForestAnomalyDetector(n_estimators=20)
    isolation_forest.fit(X_train)

    ensemble_for_scoring = FraudEnsembleModel(
        xgb_model=xgb, graphsage_model=sage, hgt_model=hgt, autoencoder=autoencoder,
        isolation_forest=isolation_forest, meta_learner=StackingMetaLearner(base_model_names=list(BASE_MODEL_ORDER)),
    )
    meta_X_train = ensemble_for_scoring.base_scores(df, homo_data, hetero_data, train_idx)
    ensemble_for_scoring.meta_learner.fit(meta_X_train, y_train)

    val_scores = ensemble_for_scoring.predict_proba(df, homo_data, hetero_data, val_idx)
    assert val_scores.shape == (len(val_idx),)
    assert np.all(val_scores >= 0) and np.all(val_scores <= 1)

    y_val = df[schema.label_col].to_numpy()[val_idx]
    calibrator = IsotonicCalibrator().fit(val_scores, y_val)
    ensemble_for_scoring.calibrator = calibrator

    calibrated_val_scores = ensemble_for_scoring.predict_proba(df, homo_data, hetero_data, val_idx)
    assert calibrated_val_scores.shape == (len(val_idx),)
    assert np.all(calibrated_val_scores >= 0) and np.all(calibrated_val_scores <= 1)
