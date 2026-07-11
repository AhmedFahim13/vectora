# tests/test_http.py
import responses

from vectora.http import PoliteSession


@responses.activate
def test_get_returns_text_and_sends_user_agent():
    responses.get("https://www.dsebd.org/test.php", body="<html>hi</html>")
    s = PoliteSession(delay_s=0)
    text = s.get("https://www.dsebd.org/test.php")
    assert text == "<html>hi</html>"
    assert "VectoraResearch" in responses.calls[0].request.headers["User-Agent"]


@responses.activate
def test_get_retries_on_5xx_then_succeeds():
    responses.get("https://www.dsebd.org/flaky.php", status=503)
    responses.get("https://www.dsebd.org/flaky.php", body="ok")
    s = PoliteSession(delay_s=0, backoff_s=0)
    assert s.get("https://www.dsebd.org/flaky.php") == "ok"
    assert len(responses.calls) == 2


@responses.activate
def test_get_raises_after_max_retries():
    import pytest
    import requests

    for _ in range(3):
        responses.get("https://www.dsebd.org/dead.php", status=500)
    s = PoliteSession(delay_s=0, backoff_s=0, max_retries=3)
    with pytest.raises(requests.HTTPError):
        s.get("https://www.dsebd.org/dead.php")
