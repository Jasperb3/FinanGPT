import unittest
from unittest import mock

import ingest


class DummyTicker:
    def __init__(self, info):
        self.info = info


class TestIngestFilters(unittest.TestCase):
    def test_etf_rejected(self) -> None:
        fake_info = {
            "quoteType": "ETF",
            "financialCurrency": "USD",
            "country": "United States",
            "isETF": True,
        }
        with mock.patch("ingest.yf.Ticker", return_value=DummyTicker(fake_info)):
            with self.assertRaises(ingest.UnsupportedInstrument):
                ingest.ensure_supported_instrument("SPY")

    def test_non_usd_rejected(self) -> None:
        fake_info = {
            "quoteType": "EQUITY",
            "financialCurrency": "EUR",
            "country": "United States",
        }
        with mock.patch("ingest.yf.Ticker", return_value=DummyTicker(fake_info)):
            with self.assertRaises(ingest.UnsupportedInstrument):
                ingest.ensure_supported_instrument("AIR.PA")


if __name__ == "__main__":
    unittest.main()
