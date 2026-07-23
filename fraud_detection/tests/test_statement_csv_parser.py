import pytest

from fraud_detection.statement_analysis.parsers.csv_parser import (
    StatementParseError,
    parse_csv_statement,
)


def test_parse_csv_with_amount_column():
    csv_bytes = (
        b"Date,Description,Amount,Balance\n"
        b"2025-01-01,Grocery Store,-45.20,954.80\n"
        b"2025-01-02,Salary,2000.00,2954.80\n"
        b"2025-01-03,Electric Bill,-120.00,2834.80\n"
    )
    df, raw_row_count = parse_csv_statement(csv_bytes)
    assert raw_row_count == 3
    assert len(df) == 3
    assert set(df.columns) >= {"date", "description", "amount", "direction", "balance"}
    assert df.loc[df["description"] == "Salary", "direction"].iloc[0] == "credit"
    assert df.loc[df["description"] == "Grocery Store", "direction"].iloc[0] == "debit"
    assert df["amount"].min() >= 0  # amounts stored as absolute value; sign lives in `direction`


def test_parse_csv_with_debit_credit_columns():
    csv_bytes = (
        b"Transaction Date,Narration,Withdrawal,Deposit,Running Balance\n"
        b"2025-02-01,ATM Withdrawal,200.00,,800.00\n"
        b"2025-02-02,Refund,,50.00,850.00\n"
    )
    df, raw_row_count = parse_csv_statement(csv_bytes)
    assert raw_row_count == 2
    assert len(df) == 2
    assert df.loc[df["description"] == "ATM Withdrawal", "direction"].iloc[0] == "debit"
    assert df.loc[df["description"] == "Refund", "direction"].iloc[0] == "credit"


def test_parse_csv_column_name_variants_are_case_insensitive():
    csv_bytes = b"DATE,PARTICULARS,amount\n2025-03-01,Test,10.00\n2025-03-02,Test2,-5.00\n"
    df, _ = parse_csv_statement(csv_bytes)
    assert len(df) == 2
    assert "description" in df.columns


def test_parse_csv_raises_on_missing_date_column():
    csv_bytes = b"Description,Amount\nGrocery,10.00\n"
    with pytest.raises(StatementParseError):
        parse_csv_statement(csv_bytes)


def test_parse_csv_raises_on_missing_amount_columns():
    csv_bytes = b"Date,Description\n2025-01-01,Grocery\n"
    with pytest.raises(StatementParseError):
        parse_csv_statement(csv_bytes)


def test_parse_csv_raises_on_empty_file():
    with pytest.raises(StatementParseError):
        parse_csv_statement(b"")


def test_parse_csv_handles_latin1_encoding():
    # café, naïve -- non-UTF-8 bytes a bank export might legitimately contain
    text = "Date,Description,Amount\n2025-01-01,Café Payment,-12.50\n"
    csv_bytes = text.encode("latin-1")
    df, _ = parse_csv_statement(csv_bytes)
    assert len(df) == 1
