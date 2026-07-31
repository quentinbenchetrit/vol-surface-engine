"""Thin client for the Deribit v2 public market-data API.

Only public endpoints are used, so no API key is required. See
https://docs.deribit.com/ for the full reference.
"""
from __future__ import annotations

import time
from typing import Any

import requests

BASE_URL = "https://www.deribit.com/api/v2"


class DeribitClient:
    def __init__(self, base_url: str = BASE_URL, timeout: float = 10.0, max_retries: int = 3):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._session = requests.Session()

    def _get(self, method: str, params: dict[str, Any]) -> Any:
        url = f"{self.base_url}/public/{method}"
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                r = self._session.get(url, params=params, timeout=self.timeout)
                r.raise_for_status()
                payload = r.json()
                if "error" in payload and payload["error"] is not None:
                    raise RuntimeError(f"Deribit error on {method}: {payload['error']}")
                return payload["result"]
            except (requests.RequestException, RuntimeError) as err:
                last_err = err
                time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"Deribit request failed after {self.max_retries} tries: {last_err}")

    def book_summary(self, currency: str = "BTC", kind: str = "option") -> list[dict]:
        """Full cross-section of a currency's book in one call.

        For options this returns every listed instrument with mark price,
        mark implied vol, best bid/ask and the per-expiry underlying (forward).
        """
        return self._get("get_book_summary_by_currency", {"currency": currency, "kind": kind})

    def instruments(self, currency: str = "BTC", kind: str = "future", expired: bool = False) -> list[dict]:
        return self._get(
            "get_instruments",
            {"currency": currency, "kind": kind, "expired": str(expired).lower()},
        )

    def index_price(self, index_name: str = "btc_usd") -> float:
        return float(self._get("get_index_price", {"index_name": index_name})["index_price"])
