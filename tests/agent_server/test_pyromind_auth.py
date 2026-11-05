import jwt
import pytest

from openhands.agent_server import pyromind_auth as auth


@pytest.fixture(autouse=True)
def clear_secret_config(monkeypatch):
    monkeypatch.delenv("PYROMIND_WEB_SECRET_KEY", raising=False)
    monkeypatch.delenv("PYROMIND_AGENT_ACCESS_KEY_REQUEST_SECRET", raising=False)
    auth.find_secret_key_v2.cache_clear()
    yield
    auth.find_secret_key_v2.cache_clear()


def test_dev_key(monkeypatch):
    monkeypatch.setenv("APP_ENV", "dev")
    assert auth.find_secret_key() == auth._DEV_SECRET_KEY


def test_prod_verifies_configured_key_and_rejects_dev_key(monkeypatch):
    monkeypatch.setenv("APP_ENV", "prod")
    monkeypatch.setenv(
        "PYROMIND_WEB_SECRET_KEY", "production-test-signing-key-32-bytes"
    )
    payload = {"sub": "tester", "email": "test@example.test", "user_id": 7}
    token = jwt.encode(payload, auth.find_secret_key(), algorithm=auth.ALGORITHM)
    assert auth.verify_jwt_token(token).user_id == 7
    dev_token = jwt.encode(payload, auth._DEV_SECRET_KEY, algorithm=auth.ALGORITHM)
    assert auth.verify_jwt_token(dev_token) is None


def test_prod_reads_mounted_secret(monkeypatch, tmp_path):
    monkeypatch.setenv("APP_ENV", "prod")
    monkeypatch.setenv("PYROMIND_SECRET_PATH", str(tmp_path))
    (tmp_path / "web_secret_key").write_text("mounted-test-key\n")
    assert auth.find_secret_key() == "mounted-test-key"


@pytest.mark.parametrize("content", [None, "\n"])
def test_prod_requires_secret(monkeypatch, tmp_path, content):
    monkeypatch.setenv("APP_ENV", "prod")
    monkeypatch.setenv("PYROMIND_SECRET_PATH", str(tmp_path))
    if content is not None:
        (tmp_path / "web_secret_key").write_text(content)
    with pytest.raises(RuntimeError):
        auth.find_secret_key()


def test_prod_reuses_agent_request_secret(monkeypatch):
    monkeypatch.setenv("APP_ENV", "prod")
    key = "shared-production-test-signing-key"
    monkeypatch.setenv("PYROMIND_AGENT_ACCESS_KEY_REQUEST_SECRET", key)
    token = jwt.encode(
        {"sub": "tester", "email": "test@example.test", "user_id": 7},
        key,
        algorithm=auth.ALGORITHM,
    )
    assert auth.verify_jwt_token(token).user_id == 7


def test_explicit_web_key_takes_precedence(monkeypatch):
    monkeypatch.setenv("PYROMIND_WEB_SECRET_KEY", "explicit-key")
    monkeypatch.setenv("PYROMIND_AGENT_ACCESS_KEY_REQUEST_SECRET", "shared-key")
    assert auth.find_secret_key() == "explicit-key"
