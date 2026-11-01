"""The local .env client forwards credentials only as request headers."""

import argparse
import json

import httpx
import pytest

from scripts import echomind_baseline as client


def test_literal_cookie_and_cluster_do_not_mutate_environment(tmp_path, monkeypatch):
    path = tmp_path / ".env.test.local"
    path.write_text(
        "PYROMIND_TEST_COOKIE='auth_token=a#$test${HOME}'\n"
        "PYROMIND_TEST_X_CLUSTER='local-cluster'\n"
    )
    monkeypatch.delenv("PYROMIND_TEST_COOKIE", raising=False)
    values, headers = client.settings(path)
    assert headers == {
        "cookie": "auth_token=a#$test${HOME}",
        "x-cluster": "local-cluster",
    }
    assert values["PYROMIND_TEST_COOKIE"] == headers["cookie"]
    import os

    assert "PYROMIND_TEST_COOKIE" not in os.environ


async def test_missing_resources_do_not_create_an_empty_conversation(
    tmp_path, monkeypatch
):
    path = tmp_path / ".env.test.local"
    path.write_text(
        "PYROMIND_TEST_COOKIE='private'\nPYROMIND_TEST_X_CLUSTER='cluster'\n"
    )
    calls = []

    async def request(*args, **kwargs):
        calls.append(args)
        return {}

    monkeypatch.setattr(client, "request", request)
    with pytest.raises(ValueError, match="model/data"):
        await client.run(argparse.Namespace(env_file=path, action="run", prompt=None))
    assert calls == []


async def test_command_body_excludes_credentials_and_identity_survives_response_loss(
    tmp_path, monkeypatch, capsys
):
    path = tmp_path / ".env.test.local"
    path.write_text(
        "PYROMIND_TEST_COOKIE='private-cookie'\nPYROMIND_TEST_X_CLUSTER='cluster'\n"
    )
    calls = []

    async def request(_http, method, resource, **kwargs):
        calls.append((method, resource, kwargs))
        if resource == client.API:
            return {"conversation_id": "accepted-conversation"}
        raise httpx.ReadTimeout("command acceptance unknown")

    monkeypatch.setattr(client, "request", request)
    with pytest.raises(httpx.ReadTimeout):
        await client.run(
            argparse.Namespace(env_file=path, action="run", prompt="Run baseline")
        )
    assert (
        json.loads(capsys.readouterr().out)["conversation_id"]
        == "accepted-conversation"
    )
    assert "private-cookie" not in json.dumps(calls)
    assert calls[0][2]["json"] == {"llm": {}}
    assert calls[1][2]["json"]["content"] == [{"type": "text", "text": "Run baseline"}]
