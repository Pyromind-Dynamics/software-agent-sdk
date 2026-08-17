import json
from pathlib import Path

import pytest
from pydantic import SecretStr

from openhands.agent_server.pyromind_llm_config import (
    LLM_CONFIG_PATH_ENV,
    LLM_FAILOVER_COOLDOWN_ENV,
    build_runtime_llm,
    load_config_entries,
    load_multimodal_config_entries,
    normalize_openai_base_url,
)
from openhands.sdk.llm import LLM, FailoverRouter


def _existing_llm() -> LLM:
    return LLM(
        model="stored-model",
        api_key=SecretStr("stored-key"),
        usage_id="conv-usage",
        num_retries=0,
    )


def _write_config(tmp_path: Path, payload) -> Path:
    path = tmp_path / "llm_config.json"
    path.write_text(json.dumps(payload))
    return path


def test_load_config_entries_object_form(monkeypatch, tmp_path):
    path = _write_config(
        tmp_path,
        {
            "llms": [
                {"name": "primary", "model": "openai/a", "base_url": "https://a"},
                {"model": "openai/b"},
            ]
        },
    )
    monkeypatch.setenv(LLM_CONFIG_PATH_ENV, str(path))

    entries = load_config_entries()
    assert [e["name"] for e in entries] == ["primary", "provider-2"]
    assert entries[0]["model"] == "openai/a"


def test_load_config_entries_bare_list_form(monkeypatch, tmp_path):
    path = _write_config(tmp_path, [{"model": "openai/a"}, {"model": "openai/b"}])
    monkeypatch.setenv(LLM_CONFIG_PATH_ENV, str(path))
    assert [e["name"] for e in load_config_entries()] == [
        "provider-1",
        "provider-2",
    ]


def test_load_config_entries_empty_when_unconfigured(monkeypatch, tmp_path):
    monkeypatch.delenv(LLM_CONFIG_PATH_ENV, raising=False)
    monkeypatch.setattr(
        "openhands.agent_server.pyromind_llm_config.default_config_path",
        lambda: tmp_path / "missing.json",
    )
    assert load_config_entries() == []


def test_load_config_entries_missing_explicit_path_raises(monkeypatch, tmp_path):
    monkeypatch.setenv(LLM_CONFIG_PATH_ENV, str(tmp_path / "missing.json"))
    with pytest.raises(FileNotFoundError):
        load_config_entries()


def test_load_config_entries_missing_default_path_ignored(monkeypatch, tmp_path):
    """An exported default path with no backing file keeps env-var mode."""
    config_path = tmp_path / "llm_config.json"
    monkeypatch.setenv(LLM_CONFIG_PATH_ENV, str(config_path))
    monkeypatch.setattr(
        "openhands.agent_server.pyromind_llm_config.default_config_path",
        lambda: config_path,
    )
    assert load_config_entries() == []


def test_load_config_entries_missing_model_raises(monkeypatch, tmp_path):
    path = _write_config(tmp_path, {"llms": [{"name": "broken"}]})
    monkeypatch.setenv(LLM_CONFIG_PATH_ENV, str(path))
    with pytest.raises(ValueError, match="model"):
        load_config_entries()


def test_load_multimodal_config_entries_absent_returns_empty(monkeypatch, tmp_path):
    path = _write_config(tmp_path, {"llms": [{"model": "openai/a"}]})
    monkeypatch.setenv(LLM_CONFIG_PATH_ENV, str(path))
    assert load_multimodal_config_entries() == []


def test_build_runtime_llm_multimodal_cohort(monkeypatch, tmp_path):
    path = _write_config(
        tmp_path,
        {
            "llms": [
                {"name": "primary", "model": "openai/a", "api_key": "key-a"},
                {"name": "fallback", "model": "openai/b", "api_key": "key-b"},
            ],
            "multimodal_llms": [
                {"name": "vision", "model": "openai/v", "api_key": "key-v"},
            ],
        },
    )
    monkeypatch.setenv(LLM_CONFIG_PATH_ENV, str(path))

    router = build_runtime_llm(_existing_llm())
    assert isinstance(router, FailoverRouter)
    assert list(router.llms_for_routing) == ["primary", "fallback"]
    assert list(router.multimodal_llms) == ["vision"]
    vision = router.multimodal_llms["vision"]
    assert vision.model == "openai/v"
    assert vision.persist_runtime_config is False
    assert vision.usage_id == "conv-usage-fallback-vision"


def test_build_runtime_llm_multimodal_name_overlap_raises(monkeypatch, tmp_path):
    path = _write_config(
        tmp_path,
        {
            "llms": [{"name": "shared", "model": "openai/a", "api_key": "key-a"}],
            "multimodal_llms": [
                {"name": "shared", "model": "openai/v", "api_key": "key-v"}
            ],
        },
    )
    monkeypatch.setenv(LLM_CONFIG_PATH_ENV, str(path))
    with pytest.raises(ValueError, match="must not overlap"):
        build_runtime_llm(_existing_llm())


def test_build_runtime_llm_multiple_returns_failover_router(monkeypatch, tmp_path):
    path = _write_config(
        tmp_path,
        {
            "llms": [
                {
                    "name": "primary",
                    "model": "openai/a",
                    "base_url": "https://a",
                    "api_key": "key-a",
                },
                {
                    "name": "fallback",
                    "model": "openai/b",
                    "base_url": "https://b",
                    "api_key": "key-b",
                },
            ]
        },
    )
    monkeypatch.setenv(LLM_CONFIG_PATH_ENV, str(path))
    monkeypatch.setenv(LLM_FAILOVER_COOLDOWN_ENV, "60")

    router = build_runtime_llm(_existing_llm())
    assert isinstance(router, FailoverRouter)
    assert list(router.llms_for_routing) == ["primary", "fallback"]
    assert router.cooldown_seconds == 60
    assert router.llms_for_routing["primary"].model == "openai/a"
    assert router.llms_for_routing["primary"].num_retries == 2
    assert router.llms_for_routing["primary"].persist_runtime_config is False
    assert (
        router.llms_for_routing["fallback"].usage_id == "conv-usage-fallback-fallback"
    )


def test_build_runtime_llm_single_provider_not_router(monkeypatch, tmp_path):
    path = _write_config(
        tmp_path,
        {"llms": [{"name": "solo", "model": "openai/a", "api_key": "key-a"}]},
    )
    monkeypatch.setenv(LLM_CONFIG_PATH_ENV, str(path))

    llm = build_runtime_llm(_existing_llm())
    assert not isinstance(llm, FailoverRouter)
    assert llm.model == "openai/a"
    assert llm.usage_id == "conv-usage"


def test_build_runtime_llm_falls_back_to_env_api_key(monkeypatch, tmp_path):
    path = _write_config(
        tmp_path,
        {
            "llms": [
                {"name": "primary", "model": "openai/a"},
                {"name": "b", "model": "openai/b"},
            ]
        },
    )
    monkeypatch.setenv(LLM_CONFIG_PATH_ENV, str(path))
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")

    router = build_runtime_llm(_existing_llm())
    assert isinstance(router, FailoverRouter)
    primary_key = router.llms_for_routing["primary"].api_key
    fallback_key = router.llms_for_routing["b"].api_key
    assert isinstance(primary_key, SecretStr)
    assert isinstance(fallback_key, SecretStr)
    assert primary_key.get_secret_value() == "env-key"
    assert fallback_key.get_secret_value() == "env-key"


def test_build_runtime_llm_api_key_env_field(monkeypatch, tmp_path):
    path = _write_config(
        tmp_path,
        {
            "llms": [
                {
                    "name": "primary",
                    "model": "openai/a",
                    "api_key_env": "OPENROUTER_API_KEY",
                },
            ]
        },
    )
    monkeypatch.setenv(LLM_CONFIG_PATH_ENV, str(path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "injected-key")

    llm = build_runtime_llm(_existing_llm())
    api_key = llm.api_key
    assert isinstance(api_key, SecretStr)
    assert api_key.get_secret_value() == "injected-key"


def test_build_runtime_llm_api_key_name_convention(monkeypatch, tmp_path):
    path = _write_config(
        tmp_path,
        {"llms": [{"name": "openrouter", "model": "openai/a"}]},
    )
    monkeypatch.setenv(LLM_CONFIG_PATH_ENV, str(path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("LLM_API_KEY_OPENROUTER", "convention-key")

    llm = build_runtime_llm(_existing_llm())
    api_key = llm.api_key
    assert isinstance(api_key, SecretStr)
    assert api_key.get_secret_value() == "convention-key"


def test_build_runtime_llm_api_key_field_beats_environment(monkeypatch, tmp_path):
    path = _write_config(
        tmp_path,
        {
            "llms": [
                {
                    "name": "primary",
                    "model": "openai/a",
                    "api_key": "config-key",
                    "api_key_env": "LLM_API_KEY_PRIMARY",
                }
            ]
        },
    )
    monkeypatch.setenv(LLM_CONFIG_PATH_ENV, str(path))
    monkeypatch.setenv("LLM_API_KEY_PRIMARY", "env-key")

    llm = build_runtime_llm(_existing_llm())
    api_key = llm.api_key
    assert isinstance(api_key, SecretStr)
    assert api_key.get_secret_value() == "config-key"


def test_build_runtime_llm_env_fallback_when_no_config_file(monkeypatch, tmp_path):
    monkeypatch.delenv(LLM_CONFIG_PATH_ENV, raising=False)
    monkeypatch.setattr(
        "openhands.agent_server.pyromind_llm_config.default_config_path",
        lambda: tmp_path / "missing.json",
    )
    monkeypatch.setenv("LLM_MODEL", "openai/env-model")
    monkeypatch.setenv("LLM_BASE_URL", "https://env.example/v1/chat/completions")
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")

    llm = build_runtime_llm(_existing_llm())
    assert llm.model == "openai/env-model"
    assert llm.base_url == "https://env.example/v1"
    api_key = llm.api_key
    assert isinstance(api_key, SecretStr)
    assert api_key.get_secret_value() == "env-key"


def test_normalize_openai_base_url_strips_chat_completions_suffix():
    assert normalize_openai_base_url("https://x/v1/chat/completions") == "https://x/v1"
    assert normalize_openai_base_url("  https://x/v1  ") == "https://x/v1"
    assert normalize_openai_base_url(None) is None
