import datetime
import unittest

import pandas as pd

from transform import prepare_dataframe


class TestTransformSchema(unittest.TestCase):
    def test_numeric_only_schema(self) -> None:
        documents = [
            {
                "ticker": "AAPL",
                "date": "2023-12-31T21:00:00+00:00",
                "payload": {
                    "income_statement": {"netIncome": 100, "note": "ignore me"},
                    "balance_sheet": {"totalAssets": 400.5, "currency": "USD"},
                    "cash_flow": {"cashAndCashEquivalents": 50},
                },
            },
            {
                "ticker": "MSFT",
                "date": "2023-12-31T21:00:00+00:00",
                "payload": {
                    "income_statement": {"netIncome": 200},
                    "balance_sheet": {},
                    "cash_flow": {},
                },
            },
        ]
        frame = prepare_dataframe(documents)
        expected_columns = ["ticker", "date", "cashAndCashEquivalents", "netIncome", "totalAssets"]
        self.assertEqual(frame.columns.tolist(), expected_columns)
        self.assertIsInstance(frame["date"].iloc[0], datetime.date)
        numeric_columns = frame.select_dtypes(include=["float64", "float32", "int64"]).columns.tolist()
        self.assertEqual(sorted(numeric_columns), sorted(expected_columns[2:]))


if __name__ == "__main__":
    unittest.main()
