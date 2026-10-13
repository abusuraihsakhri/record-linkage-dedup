"""Authorization and network-binding regression tests."""
import pytest
from fastapi import HTTPException

from agents.api import _require_token
from cli import main


def test_optional_api_bearer_token(monkeypatch):
    monkeypatch.setenv("API_TOKEN", "a-long-random-example")
    with pytest.raises(HTTPException) as error:
        _require_token(None)
    assert error.value.status_code == 401
    with pytest.raises(HTTPException):
        _require_token("Bearer wrong-token")
    assert _require_token("Bearer a-long-random-example") is None
    monkeypatch.delenv("API_TOKEN")
    assert _require_token(None) is None


def test_no_public_listen_without_token(monkeypatch):
    monkeypatch.delenv("API_TOKEN", raising=False)
    assert main(["serve", "--host", "0.0.0.0"]) == 2
