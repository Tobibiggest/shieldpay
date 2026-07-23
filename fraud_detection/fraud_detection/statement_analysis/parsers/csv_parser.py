"""Normalizes bank-statement CSV exports (column names vary widely by bank)
into a common shape: `date, description, amount, direction, balance`. This
is the shape every downstream statement-analysis module (`patterns.py`,
`anomaly_scoring.py`) is written against.
"""

from io import BytesIO
from typing import List, Optional, Tuple

import pandas as pd

DATE_ALIASES = ["date", "transaction date", "posting date", "value date", "txn date"]
DESCRIPTION_ALIASES = ["description", "narration", "particulars", "details", "memo", "transaction details"]
AMOUNT_ALIASES = ["amount", "transaction amount"]
DEBIT_ALIASES = ["debit", "withdrawal", "withdrawal amt", "money out", "debit amount"]
CREDIT_ALIASES = ["credit", "deposit", "deposit amt", "money in", "credit amount"]
BALANCE_ALIASES = ["balance", "running balance", "closing balance", "available balance"]


class StatementParseError(Exception):
    """Raised when a statement file cannot be parsed into transactions."""


def _find_column(columns: List[str], aliases: List[str]) -> Optional[str]:
    normalized = {c.strip().lower(): c for c in columns}
    for alias in aliases:
        if alias in normalized:
            return normalized[alias]
    for col_lower, col_original in normalized.items():
        for alias in aliases:
            if alias in col_lower:
                return col_original
    return None


def _read_csv_bytes(file_bytes: bytes) -> pd.DataFrame:
    if not file_bytes.strip():
        raise StatementParseError("CSV file is empty.")

    # Bank CSV exports are frequently Windows-1252/latin-1, not UTF-8.
    for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            return pd.read_csv(BytesIO(file_bytes), encoding=encoding)
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
        except pd.errors.EmptyDataError:
            raise StatementParseError("CSV file has no columns to parse.")
    raise StatementParseError("Could not parse CSV file (unsupported encoding or malformed CSV).")


def parse_csv_statement(file_bytes: bytes) -> Tuple[pd.DataFrame, int]:
    """Returns (normalized_df, raw_row_count) -- raw_row_count is the row
    count before dropping unparseable rows, so callers can report how many
    rows were lost during parsing as a data-quality signal."""
    raw = _read_csv_bytes(file_bytes)
    if raw.empty:
        raise StatementParseError("CSV file has no rows.")
    raw_row_count = len(raw)

    columns = list(raw.columns)
    date_col = _find_column(columns, DATE_ALIASES)
    description_col = _find_column(columns, DESCRIPTION_ALIASES)
    balance_col = _find_column(columns, BALANCE_ALIASES)

    if date_col is None:
        raise StatementParseError("Could not find a date column in the CSV.")

    amount_col = _find_column(columns, AMOUNT_ALIASES)
    debit_col = _find_column(columns, DEBIT_ALIASES)
    credit_col = _find_column(columns, CREDIT_ALIASES)

    df = pd.DataFrame()
    df["date"] = pd.to_datetime(raw[date_col], errors="coerce")
    df["description"] = raw[description_col].astype(str) if description_col else ""

    if amount_col is not None:
        amounts = pd.to_numeric(raw[amount_col], errors="coerce")
        df["amount"] = amounts.abs()
        df["direction"] = amounts.apply(lambda v: "credit" if v is not None and v >= 0 else "debit")
    elif debit_col is not None or credit_col is not None:
        debit = pd.to_numeric(raw[debit_col], errors="coerce").fillna(0) if debit_col else pd.Series(0, index=raw.index)
        credit = pd.to_numeric(raw[credit_col], errors="coerce").fillna(0) if credit_col else pd.Series(0, index=raw.index)
        df["amount"] = (debit + credit).abs()
        df["direction"] = ["debit" if d > 0 else "credit" for d in debit]
    else:
        raise StatementParseError("Could not find an amount, debit, or credit column in the CSV.")

    df["balance"] = pd.to_numeric(raw[balance_col], errors="coerce") if balance_col else None

    df = df.dropna(subset=["date", "amount"]).reset_index(drop=True)
    if df.empty:
        raise StatementParseError("No valid transaction rows found after parsing.")

    df = df.sort_values("date").reset_index(drop=True)
    return df, raw_row_count
