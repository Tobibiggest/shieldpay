"""Combines the base models (XGBoost, GraphSAGE, HGT, autoencoder,
IsolationForest) into one calibrated fraud probability via the stacking
meta-learner + calibrator.

GraphSAGE and HGT are transductive over a fixed graph -- they can't score a
brand-new transaction in true isolation, only a specific node index within a
graph they've already seen. So `predict_proba` here takes the prebuilt
homogeneous/heterogeneous graphs plus the row positions to score, rather than
a raw new dataframe alone. In production this implies periodically rebuilding
the graph (e.g. hourly/nightly) and scoring newly arrived transactions
against the latest snapshot -- a standard pattern for graph-based fraud
systems, not true one-row-at-a-time online graph insertion.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from torch_geometric.data import Data, HeteroData

from ..anomaly.autoencoder import TabularAutoencoder
from ..anomaly.isolation_forest import IsolationForestAnomalyDetector
from ..gnn.sage import GraphSAGEFraudModel
from ..hgnn.hgt import HGTFraudModel
from ..tabular.gbdt import XGBoostFraudModel
from .calibration import IsotonicCalibrator
from .stacking import StackingMetaLearner

BASE_MODEL_ORDER = ["xgboost", "graphsage", "hgt", "autoencoder", "isolation_forest"]


@dataclass
class FraudEnsembleModel:
    xgb_model: XGBoostFraudModel
    graphsage_model: GraphSAGEFraudModel
    hgt_model: HGTFraudModel
    autoencoder: TabularAutoencoder
    isolation_forest: IsolationForestAnomalyDetector
    meta_learner: StackingMetaLearner
    calibrator: Optional[IsotonicCalibrator] = None

    def base_scores(
        self,
        df: pd.DataFrame,
        homo_data: Data,
        hetero_data: HeteroData,
        positions: np.ndarray,
    ) -> np.ndarray:
        """Returns an [n_positions, 5] array of base-model scores, columns in
        `BASE_MODEL_ORDER` order."""
        rows = df.iloc[positions]
        X = self.xgb_model.preprocessor.transform(rows)

        xgb_scores = self.xgb_model.predict_proba(rows)
        sage_scores = self.graphsage_model.predict_proba(homo_data.x, homo_data.edge_index).numpy()[positions]
        hgt_scores = self.hgt_model.predict_proba(hetero_data).numpy()[positions]
        ae_scores = self.autoencoder.anomaly_score(X)
        if_scores = self.isolation_forest.anomaly_score(X)

        return np.stack([xgb_scores, sage_scores, hgt_scores, ae_scores, if_scores], axis=1)

    def predict_proba(
        self,
        df: pd.DataFrame,
        homo_data: Data,
        hetero_data: HeteroData,
        positions: np.ndarray,
    ) -> np.ndarray:
        meta_X = self.base_scores(df, homo_data, hetero_data, positions)
        raw = self.meta_learner.predict_proba(meta_X)
        if self.calibrator is not None:
            return self.calibrator.calibrate(raw)
        return raw
