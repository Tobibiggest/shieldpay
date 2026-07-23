from io import BytesIO
from types import SimpleNamespace

import pytest
from pypdf import PdfWriter

from fraud_detection.statement_analysis.parsers import pdf_parser
from fraud_detection.statement_analysis.parsers.csv_parser import StatementParseError
from fraud_detection.statement_analysis.parsers.pdf_parser import (
    ExtractedStatement,
    ExtractedTransaction,
)
from fraud_detection.statement_analysis.statement_analyzer import analyze_statement


def _sample_csv_bytes(n=40):
    lines = ["Date,Description,Amount,Balance"]
    balance = 1000.0
    for i in range(n):
        amount = -15.0 - i % 5
        balance += amount
        lines.append(f"2025-01-{(i % 28) + 1:02d},Merchant {i % 7},{amount:.2f},{balance:.2f}")
    return ("\n".join(lines) + "\n").encode("utf-8")


def test_analyze_statement_csv_end_to_end():
    report = analyze_statement(_sample_csv_bytes(), "statement.csv")

    assert "data_quality" in report
    assert "summary" in report
    assert "pattern_findings" in report
    assert "flagged_transactions" in report
    assert "transactions" in report
    assert report["data_quality"]["parsed_transaction_count"] == 40
    assert isinstance(report["pattern_findings"], list)
    assert len(report["transactions"]) == 40
    assert report["transactions_truncated"] is False


def test_analyze_statement_unsupported_extension_raises():
    with pytest.raises(StatementParseError):
        analyze_statement(b"not a real file", "statement.txt")


def test_analyze_statement_pdf_end_to_end(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buf = BytesIO()
    writer.write(buf)
    pdf_bytes = buf.getvalue()

    transactions = [
        ExtractedTransaction(date="2025-01-01", description="Grocery Store", amount=45.20, direction="debit", balance=954.80),
        ExtractedTransaction(date="2025-01-02", description="Salary", amount=2000.00, direction="credit", balance=2954.80),
    ]
    fake_response = SimpleNamespace(
        candidates=[SimpleNamespace(finish_reason="STOP")],
        text=ExtractedStatement(transactions=transactions, parse_confidence="high").model_dump_json(),
    )
    monkeypatch.setattr(pdf_parser, "_call_gemini_extract", lambda client, file_bytes: fake_response)

    report = analyze_statement(pdf_bytes, "statement.pdf")

    assert report["data_quality"]["parsed_transaction_count"] == 2
    assert report["data_quality"]["parse_confidence"] == "high"
    assert len(report["transactions"]) == 2
