import unittest

from src.query_engine.query import validate_sql


SCHEMA = {
    "financials.annual": ["ticker", "date", "netIncome", "totalAssets", "freeCashFlow"],
    "financials.quarterly": ["ticker", "date", "netIncome"],
}


class TestQuerySqlGuardrails(unittest.TestCase):
    def test_blocks_ddl(self) -> None:
        with self.assertRaises(ValueError):
            validate_sql("DROP TABLE financials.annual", SCHEMA)

    def test_blocks_multi_statement(self) -> None:
        with self.assertRaises(ValueError):
            validate_sql("SELECT * FROM financials.annual; DELETE FROM financials.annual", SCHEMA)

    def test_disallows_unlisted_table(self) -> None:
        with self.assertRaises(ValueError):
            validate_sql("SELECT * FROM secrets.financials", SCHEMA)

    def test_enforces_limit_cap(self) -> None:
        with self.assertRaises(ValueError):
            validate_sql("SELECT ticker FROM financials.annual LIMIT 500", SCHEMA)

    def test_default_limit_added(self) -> None:
        sql = validate_sql("SELECT ticker FROM financials.annual", SCHEMA)
        self.assertTrue(sql.lower().endswith("limit 25"))

    def test_unknown_column_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_sql("SELECT bogusColumn FROM financials.annual", SCHEMA)

    def test_cte_allowed_and_limit_respected(self) -> None:
        sql = """
        WITH ranked AS (
            SELECT ticker, freeCashFlow
            FROM financials.annual
        )
        SELECT ticker, freeCashFlow
        FROM ranked
        ORDER BY freeCashFlow DESC
        LIMIT 10
        """
        normalised = validate_sql(sql, SCHEMA)
        self.assertIn("limit 10", normalised.lower())

    def test_cte_names_not_on_allow_list_if_not_defined(self) -> None:
        sql = """
        WITH ranked AS (
            SELECT ticker FROM financials.annual
        )
        SELECT ticker FROM unknown_source
        """
        with self.assertRaises(ValueError):
            validate_sql(sql, SCHEMA)


if __name__ == "__main__":
    unittest.main()
