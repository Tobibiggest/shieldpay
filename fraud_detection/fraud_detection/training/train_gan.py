"""Trains a conditional WGAN-GP for fraud-class augmentation and compares the
downstream XGBoost baseline with vs. without the synthetic augmentation, on
the identical preprocessor and test split so the comparison is apples-to-apples.

    python -m fraud_detection.training.train_gan --data data/synthetic_relational.csv
"""

import argparse

import numpy as np

from ..data.preprocessing import fit_preprocessor
from ..evaluation.evaluate import FraudEvaluationHarness
from ..evaluation.report import write_report
from ..models.gan.cgan_wgan_gp import CWGANGPConfig, CWGANGPTrainer
from ..models.tabular.gbdt import XGBoostFraudModel
from ..utils.seed import set_global_seed
from .common import load_and_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--dataset-name", default="synthetic_relational")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--augment-multiplier",
        type=float,
        default=3.0,
        help="Synthetic fraud rows generated = multiplier * real fraud rows in the train split",
    )
    parser.add_argument("--report-dir", default="artifacts/gan")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_global_seed(args.seed)

    loaded = load_and_split(args.data, args.dataset_name)
    preprocessor = fit_preprocessor(loaded.train_df, loaded.schema)
    X_train = preprocessor.transform(loaded.train_df)
    y_train = loaded.train_df[loaded.schema.label_col].to_numpy()
    y_test = loaded.test_df[loaded.schema.label_col].to_numpy()

    harness = FraudEvaluationHarness()
    results = {}

    baseline = XGBoostFraudModel(schema=loaded.schema, preprocessor=preprocessor)
    baseline.fit_arrays(X_train, y_train)
    results["xgboost_no_augmentation"] = harness.evaluate(
        y_test, baseline.predict_proba(loaded.test_df), model_name="xgboost_no_augmentation"
    )

    print(f"Training CWGAN-GP on {len(X_train)} rows ({int(y_train.sum())} fraud)...")
    gan = CWGANGPTrainer(input_dim=X_train.shape[1], config=CWGANGPConfig(epochs=args.epochs))
    gan.fit(X_train, y_train)

    n_fraud = int(y_train.sum())
    n_synthetic = int(n_fraud * args.augment_multiplier)
    synthetic_X = gan.generate(label=1, n_samples=n_synthetic)
    synthetic_y = np.ones(n_synthetic, dtype=y_train.dtype)

    X_aug = np.concatenate([X_train, synthetic_X], axis=0)
    y_aug = np.concatenate([y_train, synthetic_y], axis=0)

    augmented = XGBoostFraudModel(schema=loaded.schema, preprocessor=preprocessor)
    augmented.fit_arrays(X_aug, y_aug)
    results["xgboost_with_cwgan_gp_augmentation"] = harness.evaluate(
        y_test, augmented.predict_proba(loaded.test_df), model_name="xgboost_with_cwgan_gp_augmentation"
    )

    print()
    print(f"Added {n_synthetic} synthetic fraud rows (real train fraud was {n_fraud})")
    print(harness.compare(results))

    for name, result in results.items():
        write_report(result, args.report_dir, run_name=name)


if __name__ == "__main__":
    main()
