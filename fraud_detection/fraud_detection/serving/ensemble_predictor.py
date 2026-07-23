"""Serving-time wrapper for the trained fraud ensemble (Phase 6) plus the
graph snapshot it was trained against, loaded once at process start (see
`training/train_ensemble.py`, which writes the bundle this loads).

GraphSAGE and HGT are transductive over a fixed graph: they can only score a
transaction that is already a node in the snapshot they were built from
(looked up here by `transaction_id`), not an arbitrary brand-new row typed in
at request time. In production this implies periodically rebuilding and
retraining against a fresh snapshot (e.g. nightly) and serving out of the
latest one -- a standard pattern for graph-based fraud systems, not
per-request live graph insertion.

This predictor intentionally does not attempt to score transactions outside
its snapshot, or remap this project's historical flat `{"features": [...]}`
payload shape onto the new schema -- that old array has a different length
and column order (from the original notebook's one-hot encoding), and
silently reinterpreting it against a differently-trained model would produce
meaningless predictions rather than an honest error. `AI_model_server_Flask/
app.py` keeps its original RandomForest path for that legacy shape unchanged,
and adds this predictor as a new, additional path for the richer payload.
"""

from pathlib import Path
from typing import List, Optional, Union

import joblib
import numpy as np


class FraudEnsemblePredictor:
    def __init__(self, ensemble, df, homo_data, hetero_data, schema):
        self.ensemble = ensemble
        self.df = df.reset_index(drop=True)
        self.homo_data = homo_data
        self.hetero_data = hetero_data
        self.schema = schema

        id_col = schema.transaction_id_col or "transaction_id"
        self._txn_id_to_position = (
            {txn_id: position for position, txn_id in enumerate(self.df[id_col])}
            if id_col in self.df.columns
            else {}
        )

    @classmethod
    def load(cls, model_dir: Union[str, Path]) -> "FraudEnsemblePredictor":
        bundle = joblib.load(Path(model_dir) / "ensemble_bundle.joblib")
        return cls(bundle["ensemble"], bundle["df"], bundle["homo_data"], bundle["hetero_data"], bundle["schema"])

    def predict_proba_by_transaction_id(self, transaction_id: str) -> Optional[float]:
        """Full graph-aware ensemble score for a transaction already present
        in the loaded snapshot. Returns None if the ID isn't in the snapshot
        (a genuinely new transaction the graph hasn't seen yet)."""
        position = self._txn_id_to_position.get(transaction_id)
        if position is None:
            return None
        scores = self.ensemble.predict_proba(self.df, self.homo_data, self.hetero_data, np.array([position]))
        return float(scores[0])

    def sample_known_transaction_ids(self, limit: int = 10) -> List[str]:
        """For smoke-testing the API without a real caller needing to know a
        valid transaction_id upfront."""
        return list(self._txn_id_to_position.keys())[:limit]
