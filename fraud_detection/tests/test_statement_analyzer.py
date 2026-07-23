import pytest

from fraud_detection.statement_analysis.parsers.csv_parser import StatementParseError
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


def test_analyze_statement_pdf_before_phase2_raises_not_implemented():
    # Phase 1 state: pdf_parser.py doesn't exist yet (added in Phase 2), so
    # this should degrade to a clear NotImplementedError rather than a
    # confusing ImportError leaking out of analyze_statement. Once Phase 2
    # adds real PDF support this test is superseded by pdf_parser-specific
    # tests and should be removed.
    with pytest.raises(NotImplementedError):
        analyze_statement(b"%PDF-1.4 fake", "statement.pdf")
