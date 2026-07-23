from io import BytesIO
from types import SimpleNamespace

import pytest
from pypdf import PdfWriter

from fraud_detection.statement_analysis.parsers import pdf_parser
from fraud_detection.statement_analysis.parsers.csv_parser import StatementParseError
from fraud_detection.statement_analysis.parsers.pdf_parser import (
    ExtractedStatement,
    ExtractedTransaction,
    parse_pdf_statement,
)


def _make_pdf_bytes(num_pages: int = 1, encrypt_password: str = None) -> bytes:
    writer = PdfWriter()
    for _ in range(num_pages):
        writer.add_blank_page(width=200, height=200)
    if encrypt_password:
        writer.encrypt(encrypt_password)
    buf = BytesIO()
    writer.write(buf)
    return buf.getvalue()


def _fake_response(transactions, parse_confidence="high", finish_reason="STOP"):
    text = None
    if transactions is not None:
        text = ExtractedStatement(
            transactions=transactions,
            parse_confidence=parse_confidence,
        ).model_dump_json()
    candidate = SimpleNamespace(finish_reason=finish_reason)
    return SimpleNamespace(candidates=[candidate], text=text)


@pytest.fixture(autouse=True)
def _fake_api_key(monkeypatch):
    # genai.Client() is constructed inside parse_pdf_statement even though
    # the real network call is monkeypatched out below -- give it a key so
    # construction doesn't fail.
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")


def test_precheck_rejects_corrupt_pdf():
    with pytest.raises(StatementParseError):
        parse_pdf_statement(b"%PDF-1.4\nnot actually a valid pdf body")


def test_precheck_rejects_encrypted_pdf():
    pdf_bytes = _make_pdf_bytes(num_pages=1, encrypt_password="secret")
    with pytest.raises(StatementParseError, match="password-protected"):
        parse_pdf_statement(pdf_bytes)


def test_precheck_rejects_too_many_pages(monkeypatch):
    monkeypatch.setattr(pdf_parser, "MAX_PDF_PAGES", 1)
    pdf_bytes = _make_pdf_bytes(num_pages=2)
    with pytest.raises(StatementParseError, match="pages"):
        parse_pdf_statement(pdf_bytes)


def test_parse_pdf_statement_success(monkeypatch):
    pdf_bytes = _make_pdf_bytes(num_pages=1)
    transactions = [
        ExtractedTransaction(date="2025-01-01", description="Grocery Store", amount=45.20, direction="debit", balance=954.80),
        ExtractedTransaction(date="2025-01-02", description="Salary", amount=2000.00, direction="credit", balance=2954.80),
    ]
    monkeypatch.setattr(
        pdf_parser, "_call_gemini_extract",
        lambda client, file_bytes: _fake_response(transactions, parse_confidence="high"),
    )

    df, raw_row_count, parse_confidence = parse_pdf_statement(pdf_bytes)

    assert raw_row_count == 2
    assert len(df) == 2
    assert parse_confidence == "high"
    assert set(df.columns) >= {"date", "description", "amount", "direction", "balance"}
    assert df["amount"].min() >= 0
    assert df.loc[df["description"] == "Salary", "direction"].iloc[0] == "credit"


def test_parse_pdf_statement_raises_on_truncated_output(monkeypatch):
    pdf_bytes = _make_pdf_bytes(num_pages=1)
    transactions = [
        ExtractedTransaction(date="2025-01-01", description="x", amount=1.0, direction="debit"),
    ]
    monkeypatch.setattr(
        pdf_parser, "_call_gemini_extract",
        lambda client, file_bytes: _fake_response(transactions, finish_reason="MAX_TOKENS"),
    )

    with pytest.raises(StatementParseError, match="truncated"):
        parse_pdf_statement(pdf_bytes)


def test_parse_pdf_statement_raises_on_no_transactions_extracted(monkeypatch):
    pdf_bytes = _make_pdf_bytes(num_pages=1)
    monkeypatch.setattr(
        pdf_parser, "_call_gemini_extract",
        lambda client, file_bytes: _fake_response([], parse_confidence="low"),
    )

    with pytest.raises(StatementParseError):
        parse_pdf_statement(pdf_bytes)


def test_parse_pdf_statement_raises_on_no_candidates(monkeypatch):
    pdf_bytes = _make_pdf_bytes(num_pages=1)
    monkeypatch.setattr(
        pdf_parser, "_call_gemini_extract",
        lambda client, file_bytes: SimpleNamespace(candidates=[], text=None),
    )

    with pytest.raises(StatementParseError, match="blocked"):
        parse_pdf_statement(pdf_bytes)
