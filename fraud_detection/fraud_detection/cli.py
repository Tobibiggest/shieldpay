"""Entry point for the fraud_detection package.

Run as `python -m fraud_detection.cli <command>` or, after an editable install
(`pip install -e .`), as `fraud-detection <command>`.

Only `generate-data` exists so far (Phase 1). `build-graph`, `train`, and
`evaluate` subcommands are added in later phases.
"""

import argparse

from .data.generators.relational_synthetic_generator import (
    generate_relational_fraud_dataset,
)


def cmd_generate_data(args: argparse.Namespace) -> None:
    df = generate_relational_fraud_dataset(
        n_transactions=args.n_transactions,
        n_accounts=args.n_accounts,
        n_devices=args.n_devices,
        n_ips=args.n_ips,
        n_merchants=args.n_merchants,
        fraud_ratio=args.fraud_ratio,
        n_fraud_rings=args.n_fraud_rings,
        seed=args.seed,
        output_csv=args.output,
    )
    print(f"Generated {len(df)} transactions -> {args.output}")
    print(f"Fraud rate: {df['label'].mean():.4f}")
    print(f"Unique senders: {df['sender_id'].nunique()}, devices: {df['device_id'].nunique()}, "
          f"ips: {df['ip_address'].nunique()}, merchants: {df['merchant_id'].nunique()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fraud_detection", description="Fraud detection training/eval CLI"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_gen = sub.add_parser("generate-data", help="Generate the relational synthetic fraud dataset")
    p_gen.add_argument("--output", default="data/synthetic_relational.csv")
    p_gen.add_argument("--n-transactions", type=int, default=20_000, dest="n_transactions")
    p_gen.add_argument("--n-accounts", type=int, default=2_000, dest="n_accounts")
    p_gen.add_argument("--n-devices", type=int, default=1_200, dest="n_devices")
    p_gen.add_argument("--n-ips", type=int, default=1_200, dest="n_ips")
    p_gen.add_argument("--n-merchants", type=int, default=300, dest="n_merchants")
    p_gen.add_argument("--fraud-ratio", type=float, default=0.03, dest="fraud_ratio")
    p_gen.add_argument("--n-fraud-rings", type=int, default=40, dest="n_fraud_rings")
    p_gen.add_argument("--seed", type=int, default=42)
    p_gen.set_defaults(func=cmd_generate_data)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
