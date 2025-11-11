from __future__ import annotations

import os

import pytest
from hypothesis import given, settings, strategies as st

from src.query_engine import query


SCHEMA = {
    "financials.annual": ["ticker", "date", "totalRevenue"],
}


SAFE_SELECTS = st.builds(
    lambda column, limit: f"SELECT {column} FROM financials.annual LIMIT {limit}",
    column=st.sampled_from(["ticker", "date", "totalRevenue", "COUNT(*)"]),
    limit=st.integers(min_value=1, max_value=100),
)


DANGEROUS = st.one_of(
    st.just("SELECT * FROM financials.annual UNION SELECT * FROM secrets"),
    st.just("SELECT * FROM financials.annual -- DROP TABLE"),
    st.just("SELECT * FROM financials.annual /* sneaky */"),
)


@given(SAFE_SELECTS)
@settings(deadline=None, max_examples=50)
def test_valid_selects_pass(sql):
    assert query.validate_sql(sql, SCHEMA)


@given(DANGEROUS)
@settings(deadline=None, max_examples=25)
def test_dangerous_patterns_blocked(sql):
    with pytest.raises(ValueError):
        query.validate_sql(sql, SCHEMA)


def test_comments_allowed_when_permitted(monkeypatch):
    monkeypatch.setenv("FINANGPT_DATA_DIR", "./data")
    monkeypatch.setattr(query, '_get_security_settings', lambda: {
        'allow_comments': True,
        'allow_union': False,
        'compat_legacy_validator': False,
    })
    sql = "SELECT ticker FROM financials.annual -- comment"
    assert query.validate_sql(sql, SCHEMA)


def test_union_allowed_when_permitted(monkeypatch):
    monkeypatch.setattr(query, '_get_security_settings', lambda: {
        'allow_comments': True,
        'allow_union': True,
        'compat_legacy_validator': False,
    })
    sql = "SELECT ticker FROM financials.annual UNION SELECT ticker FROM financials.annual"
    assert query.validate_sql(sql, SCHEMA)
