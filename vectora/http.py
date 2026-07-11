"""Polite HTTP client for dsebd.org.

verify=False is deliberate: dsebd.org serves an incomplete certificate
chain (verified 2026-07-12). Data is public; integrity risk is accepted
and the raw layer keeps checksummed copies of everything fetched.
"""
import time

import requests
import urllib3

from vectora import settings

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class PoliteSession:
    def __init__(
        self,
        delay_s: float = settings.REQUEST_DELAY_S,
        timeout_s: int = settings.REQUEST_TIMEOUT_S,
        max_retries: int = settings.MAX_RETRIES,
        backoff_s: float = 5.0,
    ):
        self.delay_s = delay_s
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self._last_request_ts = 0.0
        self._session = requests.Session()
        self._session.headers["User-Agent"] = settings.USER_AGENT
        self._session.verify = False

    def get(self, url: str, params: dict | None = None) -> str:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            self._throttle()
            try:
                resp = self._session.get(url, params=params, timeout=self.timeout_s)
                resp.raise_for_status()
                return resp.text
            except (requests.HTTPError, requests.ConnectionError, requests.Timeout) as exc:
                last_exc = exc
                time.sleep(self.backoff_s * (attempt + 1))
        raise last_exc  # type: ignore[misc]

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_ts
        if elapsed < self.delay_s:
            time.sleep(self.delay_s - elapsed)
        self._last_request_ts = time.monotonic()
