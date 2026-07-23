"""PDF bank-statement parsing via Claude's native PDF document input plus
structured outputs. Kept in its own module (not imported at package import
time) so callers that only ever use the CSV path don't need the
`anthropic`/`pypdf` dependencies -- see statement_analyzer.py's local import.
"""

import base64
from io import BytesIO
from typing import List, Literal, Optional, Tuple

import anthropic
import pandas as pd
from pydantic import BaseModel
from pypdf import PdfReader

from .csv_parser import StatementParseError

MODEL = "claude-opus-4-8"
# Non-streaming requests are capped by the SDK to ~10 minutes of estimated
# generation time (see Anthropic._calculate_nonstreaming_timeout): that works
# out to max_tokens <= 600*128000/3600 ~= 21333 for this model. Stay safely
# under that rather than switching to streaming.
MAX_OUTPUT_TOKENS = 20000
# Real constraint on extraction quality is output tokens (one JSON record per
# transaction row), not input pages -- Claude reads the whole 1M-token input
# context easily. This just guards against pathological uploads (a whole
# multi-year account history instead of one statement).
MAX_PDF_PAGES = 150


class ExtractedTransaction(BaseModel):
    date: str
    description: str
    amount: float
    direction: Literal["debit", "credit"]
    balance: Optional[float] = None


class ExtractedStatement(BaseModel):
    transactions: List[ExtractedTransaction]
    parse_confidence: Literal["high", "medium", "low"]
    notes: Optional[str] = None


EXTRACTION_PROMPT = (
    "Extract every transaction row from this bank statement PDF into structured "
    "data. For each transaction, capture: date (any format as printed), "
    "description/narration exactly as printed, amount (always positive, absolute "
    "value), direction ('debit' for money out / withdrawals, 'credit' for money "
    "in / deposits), and running balance if a balance column is present (omit if "
    "not present). Extract every transaction row in the document, in the order "
    "they appear -- do not summarize, skip, or deduplicate rows. Set "
    "parse_confidence to 'high' if the statement has a clear tabular layout "
    "extracted with full confidence, 'medium' if some rows were ambiguous or the "
    "layout was irregular, 'low' if document quality made extraction unreliable "
    "(e.g. scanned/low-resolution, handwritten, or heavily obscured). Use notes "
    "to briefly describe any ambiguity or extraction difficulty."
)


def _precheck_pdf(file_bytes: bytes) -> None:
    try:
        reader = PdfReader(BytesIO(file_bytes))
        if reader.is_encrypted:
            raise StatementParseError("PDF is password-protected; cannot extract transactions.")
        num_pages = len(reader.pages)
    except StatementParseError:
        raise
    except Exception as e:
        raise StatementParseError(f"Could not read PDF file: {e}") from e

    if num_pages == 0:
        raise StatementParseError("PDF has no pages.")
    if num_pages > MAX_PDF_PAGES:
        raise StatementParseError(
            f"PDF has {num_pages} pages; statements over {MAX_PDF_PAGES} pages are not supported."
        )


def _call_anthropic_extract(client: "anthropic.Anthropic", file_bytes: bytes):
    """Isolated so tests can monkeypatch this single function and inject a
    fake response, without needing a real API key or network access."""
    b64 = base64.standard_b64encode(file_bytes).decode("utf-8")
    return client.messages.parse(
        model=MODEL,
        max_tokens=MAX_OUTPUT_TOKENS,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": b64,
                    },
                },
                {"type": "text", "text": EXTRACTION_PROMPT},
            ],
        }],
        output_format=ExtractedStatement,
    )


def parse_pdf_statement(file_bytes: bytes) -> Tuple[pd.DataFrame, int, str]:
    """Returns (normalized_df, raw_row_count, parse_confidence) -- same
    contract as parse_csv_statement, plus the confidence Claude reported
    for its own extraction (PDF extraction, unlike CSV parsing, isn't exact)."""
    _precheck_pdf(file_bytes)

    client = anthropic.Anthropic()
    response = _call_anthropic_extract(client, file_bytes)

    if response.stop_reason == "max_tokens":
        raise StatementParseError(
            "Statement is too large to extract in one pass (output was truncated). "
            "Try splitting the PDF into smaller date ranges."
        )

    extracted = response.parsed_output
    if extracted is None or not extracted.transactions:
        raise StatementParseError("Could not extract any transactions from this PDF.")

    raw_row_count = len(extracted.transactions)
    records = [
        {
            "date": txn.date,
            "description": txn.description,
            "amount": abs(txn.amount),
            "direction": txn.direction,
            "balance": txn.balance,
        }
        for txn in extracted.transactions
    ]

    df = pd.DataFrame.from_records(records)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["date", "amount"]).reset_index(drop=True)

    if df.empty:
        raise StatementParseError("No valid transaction rows found after PDF extraction.")

    df = df.sort_values("date").reset_index(drop=True)
    return df, raw_row_count, extracted.parse_confidence
