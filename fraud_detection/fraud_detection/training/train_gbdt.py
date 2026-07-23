"""Trains the XGBoost fraud baseline and reports metrics.

    python -m fraud_detection.training.train_gbdt --data data/synthetic_relational.csv
"""

import argparse

from ..evaluation.evaluate import FraudEvaluationHarness
from ..evaluation.report import write_report
from ..models.tabular.gbdt import XGBoostFraudModel
from .common import load_and_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--dataset-name", default="synthetic_relational")
    parser.add_argument("--report-dir", default="artifacts/gbdt")
    args = parser.parse_args()

    loaded = load_and_split(args.data, args.dataset_name)

    model = XGBoostFraudModel(schema=loaded.schema).fit(loaded.train_df)

    harness = FraudEvaluationHarness()
    y_test = loaded.test_df[loaded.schema.label_col].to_numpy()
    y_score = model.predict_proba(loaded.test_df)
    result = harness.evaluate(y_test, y_score, model_name="xgboost")

    print(f"train={len(loaded.train_df)} val={len(loaded.val_df)} test={len(loaded.test_df)}")
    for k, v in result.items():
        print(f"  {k}: {v}")

    write_report(result, args.report_dir, run_name="xgboost_baseline")


if __name__ == "__main__":
    main()
