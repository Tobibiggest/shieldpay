"""Generates a synthetic fraud-transaction dataset with real relational structure.

The original generator (`AI_model_Py_Scripts/DataSetGeneratorUSingNumpy.ipynb`) emits
one flat row per transaction with no entity IDs at all -- there is no way to link
rows into a graph. This version keeps the same 20 feature signals (renamed to
snake_case) but adds `sender_id`/`recipient_id`/`device_id`/`ip_address`/
`merchant_id`/`timestamp` drawn from small, Zipf-skewed entity pools so IDs repeat
across transactions, and explicitly plants fraud rings (small groups of accounts
sharing a device/IP, transacting with a common recipient in a tight time window).
That planted structure is what gives GraphSAGE/GAT/HGT something to learn beyond
per-row features; a purely feature-driven fraud signal would make the graph models
pointless.

Feature realism is still feature-driven too (fraud rows skew toward higher amounts,
blacklisted recipients, VPN usage, etc., with substantial overlap/noise against
legitimate rows) so the GBDT/tabular baseline remains a meaningful comparison point.
"""

from datetime import timedelta
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

NUMERIC_FEATURE_COLS = [
    "transaction_amount",
    "transaction_frequency",
    "behavioral_biometrics",
    "time_since_last_transaction",
    "social_trust_score",
    "account_age",
    "normalized_transaction_amount",
    "transaction_context_anomaly",
    "fraud_complaints_count",
    "recipient_blacklist_status",
    "device_fingerprinting",
    "vpn_or_proxy_usage",
    "high_risk_transaction_time",
    "past_fraudulent_behavior_flag",
    "location_inconsistent_transaction",
    "merchant_category_mismatch",
    "user_daily_limit_exceeded",
    "recent_high_value_transaction_flag",
]
CATEGORICAL_FEATURE_COLS = ["recipient_verification_status", "geo_location_flag"]
ALL_FEATURE_COLS = NUMERIC_FEATURE_COLS + CATEGORICAL_FEATURE_COLS


def _zipf_weights(n: int, rng: np.random.Generator, alpha: float = 1.2) -> np.ndarray:
    ranks = np.arange(1, n + 1)
    weights = 1.0 / np.power(ranks, alpha)
    rng.shuffle(weights)
    return weights / weights.sum()


def _make_transaction_row(
    rng: np.random.Generator,
    sender: str,
    recipient: str,
    device: str,
    ip: str,
    merchant: str,
    timestamp: pd.Timestamp,
    is_fraud: bool,
    feature_profile: Optional[str] = None,
) -> dict:
    """`feature_profile` ("fraud" or "legit") controls which per-transaction
    feature distribution is sampled, independent of the `label` written to the
    row. It defaults to matching `is_fraud`, but planted fraud-ring
    transactions deliberately pass `feature_profile="legit"`: a ring member's
    individual transaction looks unremarkable, and only the shared
    device/IP/timing graph structure gives it away. Without this split, a
    plain tabular model reaches ~perfect separation on features alone and
    there is nothing left for GraphSAGE/HGT to add over the GBDT baseline.
    """
    if feature_profile is None:
        feature_profile = "fraud" if is_fraud else "legit"

    if feature_profile == "fraud":
        transaction_amount = rng.exponential(2500) + 500
        transaction_frequency = rng.poisson(8)
        recipient_verification_status = rng.choice(
            ["verified", "recently_registered", "suspicious"], p=[0.20, 0.35, 0.45]
        )
        recipient_blacklist_status = int(rng.random() < 0.35)
        device_fingerprinting = int(rng.random() < 0.60)
        vpn_or_proxy_usage = int(rng.random() < 0.55)
        geo_location_flag = rng.choice(["normal", "unusual"], p=[0.30, 0.70])
        behavioral_biometrics = rng.normal(0.25, 0.15)
        time_since_last_transaction = rng.exponential(5)
        social_trust_score = rng.normal(0.30, 0.15)
        account_age = rng.exponential(60)
        high_risk_transaction_time = int(rng.random() < 0.55)
        past_fraudulent_behavior_flag = int(rng.random() < 0.40)
        location_inconsistent_transaction = int(rng.random() < 0.50)
        transaction_context_anomaly = rng.normal(0.70, 0.20)
        fraud_complaints_count = rng.poisson(2)
        merchant_category_mismatch = int(rng.random() < 0.45)
        user_daily_limit_exceeded = int(rng.random() < 0.40)
        recent_high_value_transaction_flag = int(rng.random() < 0.50)
    else:
        transaction_amount = rng.exponential(150) + 5
        transaction_frequency = rng.poisson(2)
        recipient_verification_status = rng.choice(
            ["verified", "recently_registered", "suspicious"], p=[0.80, 0.17, 0.03]
        )
        recipient_blacklist_status = int(rng.random() < 0.01)
        device_fingerprinting = int(rng.random() < 0.05)
        vpn_or_proxy_usage = int(rng.random() < 0.04)
        geo_location_flag = rng.choice(["normal", "unusual"], p=[0.95, 0.05])
        behavioral_biometrics = rng.normal(0.75, 0.12)
        time_since_last_transaction = rng.exponential(48)
        social_trust_score = rng.normal(0.78, 0.12)
        account_age = rng.exponential(400) + 30
        high_risk_transaction_time = int(rng.random() < 0.08)
        past_fraudulent_behavior_flag = int(rng.random() < 0.02)
        location_inconsistent_transaction = int(rng.random() < 0.03)
        transaction_context_anomaly = rng.normal(0.15, 0.10)
        fraud_complaints_count = rng.poisson(0.05)
        merchant_category_mismatch = int(rng.random() < 0.03)
        user_daily_limit_exceeded = int(rng.random() < 0.02)
        recent_high_value_transaction_flag = int(rng.random() < 0.05)

    normalized_transaction_amount = float(np.clip(transaction_amount / 5000.0, 0, 1))
    behavioral_biometrics = float(np.clip(behavioral_biometrics, 0, 1))
    social_trust_score = float(np.clip(social_trust_score, 0, 1))
    transaction_context_anomaly = float(np.clip(transaction_context_anomaly, 0, 1))

    return {
        "sender_id": sender,
        "recipient_id": recipient,
        "device_id": device,
        "ip_address": ip,
        "merchant_id": merchant,
        "timestamp": timestamp,
        "transaction_amount": float(transaction_amount),
        "transaction_frequency": int(transaction_frequency),
        "recipient_verification_status": recipient_verification_status,
        "recipient_blacklist_status": recipient_blacklist_status,
        "device_fingerprinting": device_fingerprinting,
        "vpn_or_proxy_usage": vpn_or_proxy_usage,
        "geo_location_flag": geo_location_flag,
        "behavioral_biometrics": behavioral_biometrics,
        "time_since_last_transaction": float(max(time_since_last_transaction, 0)),
        "social_trust_score": social_trust_score,
        "account_age": float(max(account_age, 0)),
        "high_risk_transaction_time": high_risk_transaction_time,
        "past_fraudulent_behavior_flag": past_fraudulent_behavior_flag,
        "location_inconsistent_transaction": location_inconsistent_transaction,
        "normalized_transaction_amount": normalized_transaction_amount,
        "transaction_context_anomaly": transaction_context_anomaly,
        "fraud_complaints_count": int(fraud_complaints_count),
        "merchant_category_mismatch": merchant_category_mismatch,
        "user_daily_limit_exceeded": user_daily_limit_exceeded,
        "recent_high_value_transaction_flag": recent_high_value_transaction_flag,
        "label": int(is_fraud),
    }


def generate_relational_fraud_dataset(
    n_transactions: int = 20_000,
    n_accounts: int = 2_000,
    n_devices: int = 1_200,
    n_ips: int = 1_200,
    n_merchants: int = 300,
    fraud_ratio: float = 0.03,
    n_fraud_rings: int = 40,
    fraud_ring_size: Tuple[int, int] = (3, 8),
    seed: int = 42,
    start_date: str = "2025-01-01",
    end_date: str = "2025-06-30",
    output_csv: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Generate a graph-ready synthetic fraud transaction dataset.

    Returns a DataFrame sorted by `timestamp` (so downstream code can take a
    temporal train/val/test split without an extra sort), with columns matching
    `fraud_detection.data.adapters.synthetic_relational.SyntheticRelationalAdapter`'s
    schema.
    """
    rng = np.random.default_rng(seed)

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    span_seconds = (end_ts - start_ts).total_seconds()

    account_ids = np.array([f"acct_{i:06d}" for i in range(n_accounts)])
    device_ids = np.array([f"dev_{i:06d}" for i in range(n_devices)])
    ip_ids = np.array([f"ip_{i:06d}" for i in range(n_ips)])
    merchant_ids = np.array([f"mer_{i:05d}" for i in range(n_merchants)])

    account_weights = _zipf_weights(n_accounts, rng)
    device_weights = _zipf_weights(n_devices, rng)
    ip_weights = _zipf_weights(n_ips, rng)
    merchant_weights = _zipf_weights(n_merchants, rng)

    # Every account gets a "home" device + IP so shared-device/IP edges carry
    # meaning for ordinary traffic too, not only for planted fraud rings.
    home_device_idx = rng.integers(0, n_devices, size=n_accounts)
    home_ip_idx = rng.integers(0, n_ips, size=n_accounts)

    n_ring_tx_target = int(n_transactions * fraud_ratio * 0.7)

    ring_rows = []
    for _ in range(n_fraud_rings):
        ring_size = int(rng.integers(fraud_ring_size[0], fraud_ring_size[1] + 1))
        ring_members = rng.choice(account_ids, size=ring_size, replace=False)
        shared_device = rng.choice(device_ids)
        shared_ip = rng.choice(ip_ids)
        drop_account = rng.choice(account_ids)
        window_start = start_ts + timedelta(seconds=float(rng.uniform(0, max(span_seconds - 3600, 1))))
        n_ring_tx = max(ring_size, n_ring_tx_target // n_fraud_rings)
        for _ in range(n_ring_tx):
            sender = rng.choice(ring_members)
            ts = window_start + timedelta(seconds=float(rng.uniform(0, 3600)))
            # Camouflaged: ring members' individual transactions look like
            # ordinary legitimate traffic on features alone (85% of the
            # time) -- only the shared device/IP + tight time window +
            # common drop account expose them. A small residual fraction
            # keeps a weak feature signal too, since real rings aren't
            # perfectly disciplined.
            profile = "legit" if rng.random() < 0.85 else "fraud"
            ring_rows.append(
                _make_transaction_row(
                    rng,
                    sender=sender,
                    recipient=drop_account,
                    device=shared_device,
                    ip=shared_ip,
                    merchant=rng.choice(merchant_ids, p=merchant_weights),
                    timestamp=ts,
                    is_fraud=True,
                    feature_profile=profile,
                )
            )

    n_remaining = max(n_transactions - len(ring_rows), 0)
    n_background_fraud_target = max(int(n_transactions * fraud_ratio) - len(ring_rows), 0)
    background_fraud_rate = (n_background_fraud_target / n_remaining) if n_remaining else 0.0

    background_rows = []
    for _ in range(n_remaining):
        sender_idx = rng.choice(n_accounts, p=account_weights)
        sender = account_ids[sender_idx]
        recipient = rng.choice(account_ids, p=account_weights)
        if rng.random() < 0.9:
            device = device_ids[home_device_idx[sender_idx]]
            ip = ip_ids[home_ip_idx[sender_idx]]
        else:
            device = rng.choice(device_ids, p=device_weights)
            ip = rng.choice(ip_ids, p=ip_weights)
        merchant = rng.choice(merchant_ids, p=merchant_weights)
        ts = start_ts + timedelta(seconds=float(rng.uniform(0, span_seconds)))
        is_fraud = rng.random() < background_fraud_rate
        background_rows.append(
            _make_transaction_row(
                rng,
                sender=sender,
                recipient=recipient,
                device=device,
                ip=ip,
                merchant=merchant,
                timestamp=ts,
                is_fraud=bool(is_fraud),
            )
        )

    df = pd.DataFrame(ring_rows + background_rows)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df.insert(0, "transaction_id", [f"txn_{i:07d}" for i in range(len(df))])
    df = df.sort_values("timestamp").reset_index(drop=True)

    if output_csv:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

    return df
