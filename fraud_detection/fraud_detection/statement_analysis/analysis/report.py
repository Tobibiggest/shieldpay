"""Assembles the full statement-analysis result: data-quality metrics,
pattern findings, and top-K anomaly-flagged transactions, as one
JSON-serializable dict -- exactly what the Flask endpoint returns.
"""

from typing import Dict, List

import numpy as np
import pandas as pd

from .anomaly_scoring import AnomalyScoringResult, score_statement_anomalies
from .patterns import run_all_pattern_detectors

TOP_K_FLAGGED = 15
MAX_TRANSACTIONS_IN_RESPONSE = 2000  # caps JSON payload size for very large statements


def _data_quality_metrics(df: pd.DataFrame, raw_row_count: int, parse_confidence: str) -> Dict:
    exact_duplicates = int(df.duplicated(subset=["date", "description", "amount", "direction"]).sum())
    missing_description = int((df["description"].astype(str).str.strip() == "").sum())
    missing_balance = int(df["balance"].isna().sum()) if "balance" in df.columns else len(df)

    return {
        "parsed_transaction_count": len(df),
        "rows_dropped_during_parsing": max(raw_row_count - len(df), 0),
        "date_range": {
            "start": df["date"].min().isoformat() if len(df) else None,
            "end": df["date"].max().isoformat() if len(df) else None,
        },
        "exact_duplicate_rows": exact_duplicates,
        "missing_description_count": missing_description,
        "missing_balance_count": missing_balance,
        "parse_confidence": parse_confidence,
    }


def _transaction_summary(df: pd.DataFrame) -> Dict:
    debits = df.loc[df["direction"] == "debit", "amount"]
    credits = df.loc[df["direction"] == "credit", "amount"]
    return {
        "total_debit": float(debits.sum()),
        "total_credit": float(credits.sum()),
        "debit_count": int(len(debits)),
        "credit_count": int(len(credits)),
        "net_change": float(credits.sum() - debits.sum()),
    }


def _flagged_transactions(df: pd.DataFrame, anomaly: AnomalyScoringResult, top_k: int) -> List[Dict]:
    if anomaly.confidence != "ok":
        return []
    order = np.argsort(-anomaly.scores)[:top_k]
    flagged = []
    for i in order:
        row = df.iloc[i]
        flagged.append({
            "row_index": int(df.index[i]),
            "date": row["date"].isoformat(),
            "description": row["description"],
            "amount": float(row["amount"]),
            "direction": row["direction"],
            "anomaly_score": float(anomaly.scores[i]),
        })
    return flagged


def _all_transactions(df: pd.DataFrame, anomaly: AnomalyScoringResult, flagged_row_indices: set) -> List[Dict]:
    """Full (possibly truncated) transaction list, for rendering a
    timeline/table in the frontend -- `flagged_transactions` alone is only
    the top-K, not enough to chart "amounts over time"."""
    truncated = df.iloc[:MAX_TRANSACTIONS_IN_RESPONSE]
    rows = []
    for position in range(len(truncated)):
        row = truncated.iloc[position]
        row_index = int(truncated.index[position])
        rows.append({
            "row_index": row_index,
            "date": row["date"].isoformat(),
            "description": row["description"],
            "amount": float(row["amount"]),
            "direction": row["direction"],
            "anomaly_score": float(anomaly.scores[position]) if anomaly.confidence == "ok" else None,
            "flagged": row_index in flagged_row_indices,
        })
    return rows


def build_statement_report(df: pd.DataFrame, raw_row_count: int, parse_confidence: str = "high") -> Dict:
    findings = run_all_pattern_detectors(df)
    anomaly = score_statement_anomalies(df)
    flagged = _flagged_transactions(df, anomaly, TOP_K_FLAGGED)
    flagged_row_indices = {f["row_index"] for f in flagged}
    transactions = _all_transactions(df, anomaly, flagged_row_indices)

    return {
        "data_quality": _data_quality_metrics(df, raw_row_count, parse_confidence),
        "summary": _transaction_summary(df),
        "pattern_findings": findings,
        "anomaly_scoring_confidence": anomaly.confidence,
        "flagged_transactions": flagged,
        "transactions": transactions,
        "transactions_truncated": len(df) > MAX_TRANSACTIONS_IN_RESPONSE,
    }
