"""Orchestrator: `analyze_statement(file_bytes, filename) -> dict`, dispatching
to the CSV or PDF parser by file extension, then running pattern detection +
anomaly scoring + report assembly on the normalized result.
"""

import os
from typing import Dict

from .analysis.report import build_statement_report
from .parsers.csv_parser import StatementParseError, parse_csv_statement


def analyze_statement(file_bytes: bytes, filename: str) -> Dict:
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".csv":
        df, raw_row_count = parse_csv_statement(file_bytes)
        return build_statement_report(df, raw_row_count=raw_row_count, parse_confidence="high")
    elif ext == ".pdf":
        try:
            from .parsers.pdf_parser import parse_pdf_statement  # local import: avoids requiring
            # the `google-genai`/`pypdf` deps for callers that only ever use the CSV path (Phase 1).
        except ModuleNotFoundError as e:
            raise NotImplementedError(
                "PDF statement analysis requires the 'google-genai' and 'pypdf' packages "
                "(pip install -r requirements.txt) and a GEMINI_API_KEY."
            ) from e
        df, raw_row_count, parse_confidence = parse_pdf_statement(file_bytes)
        return build_statement_report(df, raw_row_count=raw_row_count, parse_confidence=parse_confidence)
    else:
        raise StatementParseError(f"Unsupported file type: {ext}")
