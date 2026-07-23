"""Generates a synthetic AML (anti-money-laundering) transaction dataset --
the first proof domain for the generalized graph platform
(`data/graph/domain_schema.py`, `data/graph/build_domain_graph.py`).

Follows the exact camouflage discipline that made the payments generator's
graph models (GraphSAGE/HGT) actually beat the tabular baseline
(`relational_synthetic_generator.py`): planted fraud rows are 85%
feature-camouflaged (drawn from the same per-row feature distribution as
legitimate rows), detectable only via graph structure. Critically, this also
means NOT leaking graph-aggregate signals (transaction counts, hold-times)
into per-row feature columns, which would trivially re-separate fraud again
and defeat the point of the graph models.

Two AML-specific fraud structures beyond what the payments generator has:
  - Layering chains: funds passed hop-to-hop through 3-6 mule accounts,
    heavy-tailed hold time between hops, a small per-hop "fee" decay, and a
    small shared device/IP pool reused across hops (the graph signal).
  - Smurfing: 15-45 distinct low-key accounts funneling amounts just under a
    reporting threshold into one collector account, spread over several
    days -- unlike payments' rings, deliberately not compressed into a tight
    window.
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
    "counterparty_trust_score",
    "account_age",
    "normalized_transaction_amount",
    "transaction_context_anomaly",
    "sar_filed_count",
    "beneficiary_blacklist_status",
    "device_fingerprinting",
    "vpn_or_proxy_usage",
    "high_risk_transaction_time",
    "past_suspicious_activity_flags",
    "jurisdiction_risk_flag",
    "shell_company_indicator",
    "daily_limit_exceeded",
    "round_trip_transaction_flag",
]
CATEGORICAL_FEATURE_COLS = ["beneficiary_verification_status", "cross_border_flag"]
ALL_FEATURE_COLS = NUMERIC_FEATURE_COLS + CATEGORICAL_FEATURE_COLS

# Common regulatory reporting thresholds -- smurfing amounts cluster just
# under one of these, mirroring statement_analysis's structuring detector.
COMMON_REPORTING_THRESHOLDS = [10_000, 50_000]


def _zipf_weights(n: int, rng: np.random.Generator, alpha: float = 1.2) -> np.ndarray:
    ranks = np.arange(1, n + 1)
    weights = 1.0 / np.power(ranks, alpha)
    rng.shuffle(weights)
    return weights / weights.sum()


def _make_transfer_row(
    rng: np.random.Generator,
    sender: str,
    recipient: str,
    device: str,
    ip: str,
    timestamp: pd.Timestamp,
    is_fraud: bool,
    feature_profile: Optional[str] = None,
    amount_override: Optional[float] = None,
) -> dict:
    """`feature_profile` controls which per-row feature distribution is
    sampled, independent of the `label` written to the row -- see the module
    docstring. `amount_override`, when given, lets structural fraud
    generators (layering, smurfing) impose a specific amount (decaying
    hop-to-hop, or just under a reporting threshold) while every OTHER
    feature still follows the profile-based camouflage split.
    """
    if feature_profile is None:
        feature_profile = "fraud" if is_fraud else "legit"

    if feature_profile == "fraud":
        transaction_amount = amount_override if amount_override is not None else (rng.exponential(4000) + 1000)
        transaction_frequency = rng.poisson(6)
        beneficiary_verification_status = rng.choice(
            ["verified", "recently_registered", "unverified"], p=[0.15, 0.35, 0.50]
        )
        beneficiary_blacklist_status = int(rng.random() < 0.30)
        device_fingerprinting = int(rng.random() < 0.55)
        vpn_or_proxy_usage = int(rng.random() < 0.50)
        cross_border_flag = rng.choice(["domestic", "cross_border"], p=[0.35, 0.65])
        behavioral_biometrics = rng.normal(0.30, 0.15)
        time_since_last_transaction = rng.exponential(4)
        counterparty_trust_score = rng.normal(0.32, 0.15)
        account_age = rng.exponential(50)
        high_risk_transaction_time = int(rng.random() < 0.50)
        past_suspicious_activity_flags = int(rng.random() < 0.35)
        jurisdiction_risk_flag = int(rng.random() < 0.40)
        transaction_context_anomaly = rng.normal(0.65, 0.20)
        sar_filed_count = rng.poisson(1.5)
        shell_company_indicator = int(rng.random() < 0.30)
        daily_limit_exceeded = int(rng.random() < 0.35)
        round_trip_transaction_flag = int(rng.random() < 0.20)
    else:
        # Camouflaged: ignore amount_override here even though it was passed
        # by a layering/smurfing caller -- the whole point of the 85% "legit"
        # profile split is that these rows draw from the SAME distribution as
        # ordinary legit rows, amount included. If the structural (decaying /
        # near-threshold) amount leaked through here too, "large transaction"
        # alone would trivially separate the fraud -- exactly the mistake
        # made building the payments generator (see module docstring).
        if rng.random() < 0.12:
            # a minority of ordinary legit traffic is business-scale (invoices,
            # rent, real estate deposits, etc.), giving genuine overlap with
            # the fraud amount range rather than a hard gap.
            transaction_amount = rng.exponential(3000) + 800
        else:
            transaction_amount = rng.exponential(300) + 20
        transaction_frequency = rng.poisson(2)
        beneficiary_verification_status = rng.choice(
            ["verified", "recently_registered", "unverified"], p=[0.82, 0.15, 0.03]
        )
        beneficiary_blacklist_status = int(rng.random() < 0.01)
        device_fingerprinting = int(rng.random() < 0.05)
        vpn_or_proxy_usage = int(rng.random() < 0.04)
        cross_border_flag = rng.choice(["domestic", "cross_border"], p=[0.85, 0.15])
        behavioral_biometrics = rng.normal(0.76, 0.12)
        time_since_last_transaction = rng.exponential(50)
        counterparty_trust_score = rng.normal(0.80, 0.12)
        account_age = rng.exponential(420) + 30
        high_risk_transaction_time = int(rng.random() < 0.07)
        past_suspicious_activity_flags = int(rng.random() < 0.02)
        jurisdiction_risk_flag = int(rng.random() < 0.05)
        transaction_context_anomaly = rng.normal(0.14, 0.10)
        sar_filed_count = rng.poisson(0.03)
        shell_company_indicator = int(rng.random() < 0.02)
        daily_limit_exceeded = int(rng.random() < 0.02)
        round_trip_transaction_flag = int(rng.random() < 0.01)

    normalized_transaction_amount = float(np.clip(transaction_amount / 50_000.0, 0, 1))
    behavioral_biometrics = float(np.clip(behavioral_biometrics, 0, 1))
    counterparty_trust_score = float(np.clip(counterparty_trust_score, 0, 1))
    transaction_context_anomaly = float(np.clip(transaction_context_anomaly, 0, 1))

    return {
        "sender_id": sender,
        "recipient_id": recipient,
        "device_id": device,
        "ip_address": ip,
        "timestamp": timestamp,
        "transaction_amount": float(transaction_amount),
        "transaction_frequency": int(transaction_frequency),
        "beneficiary_verification_status": beneficiary_verification_status,
        "beneficiary_blacklist_status": beneficiary_blacklist_status,
        "device_fingerprinting": device_fingerprinting,
        "vpn_or_proxy_usage": vpn_or_proxy_usage,
        "cross_border_flag": cross_border_flag,
        "behavioral_biometrics": behavioral_biometrics,
        "time_since_last_transaction": float(max(time_since_last_transaction, 0)),
        "counterparty_trust_score": counterparty_trust_score,
        "account_age": float(max(account_age, 0)),
        "high_risk_transaction_time": high_risk_transaction_time,
        "past_suspicious_activity_flags": past_suspicious_activity_flags,
        "jurisdiction_risk_flag": jurisdiction_risk_flag,
        "normalized_transaction_amount": normalized_transaction_amount,
        "transaction_context_anomaly": transaction_context_anomaly,
        "sar_filed_count": int(sar_filed_count),
        "shell_company_indicator": shell_company_indicator,
        "daily_limit_exceeded": daily_limit_exceeded,
        "round_trip_transaction_flag": round_trip_transaction_flag,
        "label": int(is_fraud),
    }


def generate_aml_transaction_dataset(
    n_transactions: int = 20_000,
    n_accounts: int = 2_000,
    n_devices: int = 1_200,
    n_ips: int = 1_200,
    fraud_ratio: float = 0.03,
    n_layering_chains: int = 25,
    layering_chain_len: Tuple[int, int] = (3, 6),
    n_smurf_rings: int = 15,
    smurf_ring_size: Tuple[int, int] = (15, 45),
    seed: int = 42,
    start_date: str = "2025-01-01",
    end_date: str = "2025-06-30",
    output_csv: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Generate a graph-ready synthetic AML transaction dataset. Returns a
    DataFrame sorted by `timestamp`, with columns matching
    `fraud_detection.domains.aml.aml_adapter.AMLAdapter`'s schema.
    """
    rng = np.random.default_rng(seed)

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    span_seconds = (end_ts - start_ts).total_seconds()

    account_ids = np.array([f"acct_{i:06d}" for i in range(n_accounts)])
    device_ids = np.array([f"dev_{i:06d}" for i in range(n_devices)])
    ip_ids = np.array([f"ip_{i:06d}" for i in range(n_ips)])

    account_weights = _zipf_weights(n_accounts, rng)
    device_weights = _zipf_weights(n_devices, rng)
    ip_weights = _zipf_weights(n_ips, rng)

    home_device_idx = rng.integers(0, n_devices, size=n_accounts)
    home_ip_idx = rng.integers(0, n_ips, size=n_accounts)

    # ---- Layering chains: funds hop through a sequence of mule accounts ----
    layering_rows = []
    for _ in range(n_layering_chains):
        chain_len = int(rng.integers(layering_chain_len[0], layering_chain_len[1] + 1))
        mules = rng.choice(account_ids, size=chain_len + 1, replace=False)
        shared_devices = rng.choice(device_ids, size=int(rng.integers(1, 3)), replace=False)
        shared_ips = rng.choice(ip_ids, size=int(rng.integers(1, 3)), replace=False)
        fee_rate = rng.uniform(0.01, 0.05)
        amount = rng.exponential(6000) + 1500
        window_start = start_ts + timedelta(seconds=float(rng.uniform(0, max(span_seconds - 7 * 86400, 1))))

        ts = window_start
        for hop in range(chain_len):
            ts = ts + timedelta(seconds=float(rng.exponential(3 * 3600)))  # heavy-tailed hold time, no hard cutoff
            amount *= (1 - fee_rate)
            profile = "legit" if rng.random() < 0.85 else "fraud"
            layering_rows.append(
                _make_transfer_row(
                    rng,
                    sender=mules[hop],
                    recipient=mules[hop + 1],
                    device=rng.choice(shared_devices),
                    ip=rng.choice(shared_ips),
                    timestamp=ts,
                    is_fraud=True,
                    feature_profile=profile,
                    amount_override=float(amount),
                )
            )

    # ---- Smurfing: many small originators funnel into one collector ----
    smurf_rows = []
    for _ in range(n_smurf_rings):
        collector = rng.choice(account_ids)
        n_smurfs = int(rng.integers(smurf_ring_size[0], smurf_ring_size[1] + 1))
        smurfs = rng.choice(account_ids, size=n_smurfs, replace=False)
        window_start = start_ts + timedelta(seconds=float(rng.uniform(0, max(span_seconds - 7 * 86400, 1))))
        threshold = rng.choice(COMMON_REPORTING_THRESHOLDS)

        for smurf in smurfs:
            amt = rng.uniform(0.70, 0.97) * threshold
            ts = window_start + timedelta(hours=float(rng.uniform(0, 24 * 5)))
            profile = "legit" if rng.random() < 0.85 else "fraud"
            smurf_rows.append(
                _make_transfer_row(
                    rng,
                    sender=smurf,
                    recipient=collector,
                    device=rng.choice(device_ids),  # smurfs use their own varied devices, unlike layering mules
                    ip=rng.choice(ip_ids),
                    timestamp=ts,
                    is_fraud=True,
                    feature_profile=profile,
                    amount_override=float(amt),
                )
            )

    # ---- Background legitimate traffic + a small amount of scattered, feature-obvious fraud ----
    n_structural_fraud = len(layering_rows) + len(smurf_rows)
    n_remaining = max(n_transactions - n_structural_fraud, 0)
    n_background_fraud_target = max(int(n_transactions * fraud_ratio) - n_structural_fraud, 0)
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
        ts = start_ts + timedelta(seconds=float(rng.uniform(0, span_seconds)))
        is_fraud = rng.random() < background_fraud_rate
        background_rows.append(
            _make_transfer_row(
                rng, sender=sender, recipient=recipient, device=device, ip=ip, timestamp=ts, is_fraud=bool(is_fraud)
            )
        )

    df = pd.DataFrame(layering_rows + smurf_rows + background_rows)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df.insert(0, "transaction_id", [f"txn_{i:07d}" for i in range(len(df))])
    df = df.sort_values("timestamp").reset_index(drop=True)

    if output_csv:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

    return df
