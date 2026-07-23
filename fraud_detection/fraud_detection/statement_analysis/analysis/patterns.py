"""Rule-based, no-ML pattern detectors over a normalized statement DataFrame
(columns: date, description, amount, direction, balance). Each detector
returns a list of finding dicts: {pattern_type, description, severity,
affected_row_indices}. These run independently of, and in addition to, the
ML anomaly scoring in `anomaly_scoring.py` -- they name structurally
specific patterns (Benford's law, structuring, duplicates) that a generic
anomaly score wouldn't call out explicitly.
"""

from typing import Dict, List

import numpy as np
import pandas as pd

SEVERITY_LOW, SEVERITY_MEDIUM, SEVERITY_HIGH = "low", "medium", "high"

BENFORD_EXPECTED = {d: np.log10(1 + 1 / d) for d in range(1, 10)}

# Common regulatory reporting thresholds across several currencies/magnitudes
# -- a generic proxy since the statement's currency isn't reliably known.
COMMON_REPORTING_THRESHOLDS = [10_000, 50_000, 100_000, 500_000, 1_000_000]


def _finding(pattern_type: str, description: str, severity: str, affected_row_indices: List[int]) -> Dict:
    return {
        "pattern_type": pattern_type,
        "description": description,
        "severity": severity,
        "affected_row_indices": affected_row_indices,
    }


def detect_velocity_spikes(df: pd.DataFrame) -> List[Dict]:
    findings = []
    daily = df.groupby(df["date"].dt.date).agg(count=("amount", "size"), volume=("amount", "sum"))
    if len(daily) < 3:
        return findings
    count_mean, count_std = daily["count"].mean(), daily["count"].std()
    volume_mean, volume_std = daily["volume"].mean(), daily["volume"].std()
    for day, row in daily.iterrows():
        if count_std > 0 and (row["count"] - count_mean) / count_std > 2.5:
            idx = df.index[df["date"].dt.date == day].tolist()
            findings.append(_finding(
                "velocity_spike",
                f"{row['count']:.0f} transactions on {day} is far above this statement's typical day "
                f"({count_mean:.1f} average).",
                SEVERITY_MEDIUM, idx,
            ))
        elif volume_std > 0 and (row["volume"] - volume_mean) / volume_std > 2.5:
            idx = df.index[df["date"].dt.date == day].tolist()
            findings.append(_finding(
                "velocity_spike",
                f"Transaction volume on {day} ({row['volume']:.2f}) is far above this statement's typical day.",
                SEVERITY_MEDIUM, idx,
            ))
    return findings


def detect_benfords_law_deviation(df: pd.DataFrame) -> List[Dict]:
    amounts = df.loc[df["amount"] > 0, "amount"]
    if len(amounts) < 50:
        return []

    def leading_digit(a: float) -> int:
        s = f"{a:.10f}".lstrip("0.").replace(".", "")
        return int(s[0]) if s else 1

    leading_digits = amounts.apply(leading_digit)
    observed = leading_digits.value_counts(normalize=True).reindex(range(1, 10), fill_value=0)
    deviation = sum(abs(observed[d] - BENFORD_EXPECTED[d]) for d in range(1, 10))
    if deviation > 0.25:  # empirical threshold: organic financial data rarely exceeds ~0.1-0.15
        return [_finding(
            "benfords_law_deviation",
            f"Leading-digit distribution of transaction amounts deviates from Benford's Law "
            f"(deviation score {deviation:.2f}), which can indicate fabricated or manipulated figures.",
            SEVERITY_MEDIUM, [],
        )]
    return []


def detect_round_number_bias(df: pd.DataFrame) -> List[Dict]:
    if len(df) < 10:
        return []
    is_round = (df["amount"] % 100 == 0) & (df["amount"] > 0)
    ratio = is_round.mean()
    if ratio > 0.3:
        idx = df.index[is_round].tolist()
        return [_finding(
            "round_number_bias",
            f"{ratio:.0%} of transactions are round hundreds, notably higher than typical organic "
            f"spending patterns.",
            SEVERITY_LOW, idx,
        )]
    return []


def detect_structuring(df: pd.DataFrame) -> List[Dict]:
    findings = []
    for threshold in COMMON_REPORTING_THRESHOLDS:
        near = df[(df["amount"] >= threshold * 0.85) & (df["amount"] < threshold)]
        if len(near) >= 3:
            span_days = (near["date"].max() - near["date"].min()).days
            if span_days <= 14:
                findings.append(_finding(
                    "structuring",
                    f"{len(near)} transactions just under {threshold:,.0f} within {span_days} days -- "
                    f"a pattern consistent with structuring to avoid a reporting threshold.",
                    SEVERITY_HIGH, near.index.tolist(),
                ))
    return findings


def detect_duplicate_transactions(df: pd.DataFrame) -> List[Dict]:
    findings = []
    for (_amount, _description), group in df.groupby(["amount", "description"]):
        if len(group) < 2:
            continue
        gaps = group["date"].sort_values().diff().dt.days.dropna()
        if (gaps <= 2).any():
            findings.append(_finding(
                "duplicate_transaction",
                f"{len(group)} transactions of the same amount and description within 2 days of "
                f"each other.",
                SEVERITY_MEDIUM, group.index.tolist(),
            ))
    return findings


def detect_dormant_then_burst(df: pd.DataFrame) -> List[Dict]:
    findings = []
    if len(df) < 5:
        return findings
    sorted_df = df.sort_values("date").reset_index()
    gaps = sorted_df["date"].diff().dt.days
    for i in range(1, len(sorted_df)):
        if pd.notna(gaps.iloc[i]) and gaps.iloc[i] >= 30:
            burst_window = sorted_df.iloc[i:i + 5]
            burst_span = (burst_window["date"].max() - burst_window["date"].min()).days
            if len(burst_window) >= 3 and burst_span <= 3:
                findings.append(_finding(
                    "dormant_then_burst",
                    f"{gaps.iloc[i]:.0f} days of inactivity followed by a burst of "
                    f"{len(burst_window)} transactions within {burst_span} days.",
                    SEVERITY_MEDIUM, burst_window["index"].tolist(),
                ))
    return findings


def detect_recipient_concentration(df: pd.DataFrame) -> List[Dict]:
    findings = []
    debits = df[df["direction"] == "debit"]
    if debits.empty:
        return findings

    by_recipient = debits.groupby("description")["amount"].sum().sort_values(ascending=False)
    total = by_recipient.sum()
    if total > 0 and len(by_recipient) > 1:
        top_recipient, top_amount = by_recipient.index[0], by_recipient.iloc[0]
        share = top_amount / total
        if share > 0.4:
            idx = debits.index[debits["description"] == top_recipient].tolist()
            findings.append(_finding(
                "recipient_concentration",
                f"'{top_recipient}' accounts for {share:.0%} of all outgoing funds in this statement.",
                SEVERITY_MEDIUM, idx,
            ))

    typical_amount = debits["amount"].median()
    counts = debits["description"].value_counts()
    first_time_large = debits[
        (debits["description"].map(counts) == 1) & (debits["amount"] > typical_amount * 5)
    ]
    if not first_time_large.empty and typical_amount > 0:
        findings.append(_finding(
            "first_time_large_recipient",
            f"{len(first_time_large)} transaction(s) to a recipient seen only once, for an amount "
            f"well above this statement's typical transaction size.",
            SEVERITY_MEDIUM, first_time_large.index.tolist(),
        ))
    return findings


def detect_amount_outliers(df: pd.DataFrame) -> List[Dict]:
    if len(df) < 10:
        return []
    q1, q3 = df["amount"].quantile(0.25), df["amount"].quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return []
    upper_bound = q3 + 3 * iqr
    outliers = df[df["amount"] > upper_bound]
    if outliers.empty:
        return []
    return [_finding(
        "amount_outlier",
        f"{len(outliers)} transaction(s) far larger than the rest of this statement's typical amounts.",
        SEVERITY_LOW, outliers.index.tolist(),
    )]


ALL_DETECTORS = [
    detect_velocity_spikes,
    detect_benfords_law_deviation,
    detect_round_number_bias,
    detect_structuring,
    detect_duplicate_transactions,
    detect_dormant_then_burst,
    detect_recipient_concentration,
    detect_amount_outliers,
]


def run_all_pattern_detectors(df: pd.DataFrame) -> List[Dict]:
    findings = []
    for detector in ALL_DETECTORS:
        findings.extend(detector(df))
    return findings
