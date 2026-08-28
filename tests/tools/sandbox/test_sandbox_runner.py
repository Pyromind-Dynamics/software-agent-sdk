"""Tests for the frozen runtime ``sandbox_runner.py``.

The runner lives in the skill's ``scripts/`` directory (it is a standalone
CLI, not a package), so the tests load it via ``importlib`` with a mocked
``SandboxClient`` — no platform calls are made.
"""

import importlib.util
import json
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from openhands.sdk.profiles import ProcessingProfile, ProcessingStep


_RUNNER_PATH = (
    Path(__file__).resolve().parents[3]
    / ".agents"
    / "skills"
    / "data-processing"
    / "scripts"
    / "edp"
    / "sandbox_runner.py"
)


def _load_runner() -> ModuleType:
    cached = sys.modules.get("sandbox_runner")
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location("sandbox_runner", _RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["sandbox_runner"] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


def _profile(**overrides: Any) -> ProcessingProfile:
    payload: dict[str, Any] = {
        "name": "test-profile",
        "steps": [
            {"name": "create_sandbox", "params": {"image": "{image}"}},
            {"name": "probe", "params": {"command": "test -d {workdir}"}},
            {
                "name": "write_file",
                "params": {"path": "/workspace/t.sh", "content_field": "test_sh"},
            },
            {"name": "exec", "params": {"command": "bash /workspace/t.sh"}},
            {"name": "delete_sandbox", "params": {}},
        ],
        "verdict": {"kind": "exit_code", "success_codes": [0]},
    }
    payload.update(overrides)
    return ProcessingProfile.model_validate(payload)


def _record(task_id: str = "t-1", image: str = "img:1") -> dict[str, Any]:
    return {
        "task_id": task_id,
        "image": image,
        "workdir": "/workspace",
        "test_sh": "echo hi",
        "prompt": "solve the task",
    }


def _mock_client(
    probe_code: int = 0, exec_code: int = 0, create_error: Exception | None = None
) -> MagicMock:
    client = MagicMock()
    if create_error is not None:
        client.create.side_effect = create_error
    else:
        client.create.return_value = SimpleNamespace(id="sb-1")
    client.wait_for_sandbox_status.return_value = True
    # The same client is reused across records: odd calls are probes, even
    # calls are the verifier exec.
    call_counter = {"n": 0}

    def _exec_command(sandbox_id: str, command: str, **kwargs: Any) -> SimpleNamespace:
        call_counter["n"] += 1
        code = probe_code if call_counter["n"] % 2 == 1 else exec_code
        return SimpleNamespace(returncode=code, output="")

    client.exec_command.side_effect = _exec_command
    return client


# ---------------------------------------------------------------------------
# decide_verdict
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("exit_code", "error_category", "expected"),
    [
        (0, None, ("usable", None)),
        (1, None, ("error", runner.CATEGORY_VERIFIER_FAILED)),
        (None, runner.CATEGORY_CREATE_FAILED, ("error", runner.CATEGORY_CREATE_FAILED)),
    ],
)
def test_decide_verdict(
    exit_code: int | None,
    error_category: str | None,
    expected: tuple[str, str | None],
) -> None:
    rule = _profile().verdict
    assert runner.decide_verdict(rule, exit_code, error_category) == expected


def test_decide_verdict_honors_custom_success_codes() -> None:
    rule = _profile(verdict={"kind": "exit_code", "success_codes": [0, 42]}).verdict
    assert runner.decide_verdict(rule, 42, None) == ("usable", None)


# ---------------------------------------------------------------------------
# validate_record
# ---------------------------------------------------------------------------


def test_validate_record_usable_and_cleans_up() -> None:
    client = _mock_client()
    entry = runner.validate_record(_profile(), _record(), client)

    assert entry.verdict == "usable"
    assert entry.exit_code == 0
    assert entry.error_category is None
    request = client.create.call_args.args[0]
    assert request.image == "img:1"
    client.delete.assert_called_once_with("sb-1")
    client.write_file.assert_called_once_with("sb-1", "/workspace/t.sh", b"echo hi")


def test_validate_record_probe_failure_still_cleans_up() -> None:
    client = _mock_client(probe_code=1)
    entry = runner.validate_record(_profile(), _record(), client)

    assert entry.verdict == "error"
    assert entry.error_category == runner.CATEGORY_PROBE_FAILED
    client.delete.assert_called_once_with("sb-1")


def test_validate_record_verifier_failure_reports_exit_code() -> None:
    client = _mock_client(exec_code=3)
    entry = runner.validate_record(_profile(), _record(), client)

    assert entry.verdict == "error"
    assert entry.exit_code == 3
    assert entry.error_category == runner.CATEGORY_VERIFIER_FAILED
    client.delete.assert_called_once_with("sb-1")


def test_validate_record_create_failure_skips_delete() -> None:
    client = _mock_client(create_error=RuntimeError("boom"))
    entry = runner.validate_record(_profile(), _record(), client)

    assert entry.verdict == "error"
    assert entry.error_category == runner.CATEGORY_CREATE_FAILED
    client.delete.assert_not_called()


def test_validate_record_delete_pauses_running_sandbox_first() -> None:
    client = _mock_client()
    client.delete.side_effect = [
        RuntimeError(
            "INTERNAL_SERVER_ERROR: InstanceService.delete_instance-"
            "instance`s status is Running, can not delete!"
        ),
        None,
    ]
    entry = runner.validate_record(_profile(), _record(), client)

    assert entry.verdict == "usable"
    client.pause.assert_called_once_with("sb-1")
    assert client.delete.call_count == 2


def test_validate_record_missing_placeholder_field_is_error() -> None:
    client = _mock_client()
    record = _record()
    del record["workdir"]
    entry = runner.validate_record(_profile(), record, client)

    assert entry.verdict == "error"
    assert entry.error_category == runner.CATEGORY_EXEC_FAILED
    client.delete.assert_called_once_with("sb-1")


def test_substitute_empty_secret_value_is_error() -> None:
    """An empty credential must fail loudly, never silently reach the CC env.

    Regression guard for the 6fc838f5 session: a credential that arrives as
    an empty string used to render as `export K=''` and left Claude Code with
    apiKeySource=none.
    """
    with pytest.raises(runner.StepError, match="not provided"):
        runner._substitute("{secret:LLM_AUTH_TOKEN}", {}, {"LLM_AUTH_TOKEN": ""})
    with pytest.raises(runner.StepError, match="not provided"):
        runner._substitute("{secret:LLM_AUTH_TOKEN}", {}, None)


def test_validate_record_duplicate_create_is_error() -> None:
    profile = _profile()
    profile.steps.insert(1, profile.steps[0])
    client = _mock_client()
    entry = runner.validate_record(profile, _record(), client)

    assert entry.verdict == "error"
    assert entry.error_category == runner.CATEGORY_CREATE_FAILED
    client.delete.assert_called_once_with("sb-1")


def test_validate_record_step_before_create_is_error() -> None:
    profile = _profile(steps=[{"name": "exec", "params": {"command": "true"}}])
    client = _mock_client()
    entry = runner.validate_record(profile, _record(), client)

    assert entry.verdict == "error"
    assert entry.error_category == runner.CATEGORY_EXEC_FAILED
    client.create.assert_not_called()


# ---------------------------------------------------------------------------
# run_batch: resume + image-dedup cache + limit
# ---------------------------------------------------------------------------


def test_run_batch_caches_verdicts_per_image(tmp_path: Path) -> None:
    client = _mock_client()
    records = [
        _record("t-1"),
        _record("t-2", image="img:1"),
        _record("t-3", image="img:2"),
    ]
    summary = runner.run_batch(
        _profile(), records, tmp_path, client, dedup_by_image=True
    )

    assert (summary.total, summary.usable, summary.cached) == (3, 3, 1)
    assert client.create.call_count == 2  # t-2 reuses img:1

    lines = (tmp_path / "verdicts.jsonl").read_text().splitlines()
    entries = [json.loads(line) for line in lines]
    assert [e["task_id"] for e in entries] == ["t-1", "t-2", "t-3"]
    assert entries[1]["cached"] is True
    assert entries[0]["cached"] is False


def test_run_batch_dedup_off_by_default_runs_every_record(tmp_path: Path) -> None:
    client = _mock_client()
    records = [_record("t-1"), _record("t-2", image="img:1")]

    summary = runner.run_batch(_profile(), records, tmp_path, client)

    # Harness runs must be judged per record even on a shared image.
    assert (summary.total, summary.usable, summary.cached) == (2, 2, 0)
    assert client.create.call_count == 2


def test_run_batch_resumes_from_existing_verdicts(tmp_path: Path) -> None:
    verdicts = tmp_path / "verdicts.jsonl"
    verdicts.write_text(
        json.dumps({"task_id": "t-1", "image": "img:1", "verdict": "usable"}) + "\n"
    )
    client = _mock_client()
    summary = runner.run_batch(
        _profile(), [_record("t-1"), _record("t-2", image="img:2")], tmp_path, client
    )

    assert (summary.total, summary.resumed, summary.usable) == (2, 1, 1)
    assert client.create.call_count == 1  # only t-2 actually ran

    entries = [json.loads(line) for line in verdicts.read_text().splitlines()]
    assert [e["task_id"] for e in entries] == ["t-1", "t-2"]


def test_run_batch_limit_stops_early(tmp_path: Path) -> None:
    client = _mock_client()
    records = [_record(f"t-{i}", image=f"img:{i}") for i in range(5)]
    summary = runner.run_batch(_profile(), records, tmp_path, client, limit=2)

    assert summary.total == 2
    assert client.create.call_count == 2


def test_run_batch_logs_one_line_per_record(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Node-log observability: exactly one [i/N] progress line per record."""
    client = _mock_client()
    records = [_record("t-1"), _record("t-2", image="img:1")]
    with caplog.at_level(logging.INFO, logger="sandbox_runner"):
        runner.run_batch(_profile(), records, tmp_path, client, dedup_by_image=True)

    progress_lines = [
        rec.message for rec in caplog.records if rec.message.startswith("[")
    ]
    assert progress_lines == [
        "[1/2] task_id=t-1 verdict=usable reward=None",
        "[2/2] task_id=t-2 verdict=usable reward=None (cached)",
    ]


def test_load_existing_verdicts_tolerates_bad_lines(tmp_path: Path) -> None:
    verdicts = tmp_path / "verdicts.jsonl"
    verdicts.write_text(
        "\n".join(
            [
                json.dumps({"task_id": "t-1", "image": "img:1", "verdict": "usable"}),
                "{not json",
                json.dumps({"no_task_id": True}),
                json.dumps(
                    {
                        "task_id": "t-2",
                        "image": "img:2",
                        "verdict": "error",
                        "cached": True,
                    }
                ),
            ]
        )
        + "\n"
    )
    completed, image_cache = runner.load_existing_verdicts(verdicts)

    assert completed == {"t-1", "t-2"}
    # Cached verdicts must not seed the image cache (only real runs do).
    assert set(image_cache) == {"img:1"}


# ---------------------------------------------------------------------------
# CLI end-to-end with injected client factory
# ---------------------------------------------------------------------------


def test_main_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(_profile().model_dump_json())
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(json.dumps(_record()) + "\n")

    client = _mock_client()
    factory_calls: list[dict[str, Any]] = []

    def factory(**kwargs: Any) -> MagicMock:
        factory_calls.append(kwargs)
        return client

    monkeypatch.delenv("PYROMIND_AUTH_TOKEN", raising=False)
    exit_code = runner.main(
        [
            "--profile",
            str(profile_path),
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(tmp_path / "run"),
            "--env",
            "pre",
            "--cluster",
            "us-west-1",
            "--auth-token",
            "tok",
        ],
        client_factory=factory,
    )

    assert exit_code == 0
    assert factory_calls == [
        {"env": "pre", "cluster": "us-west-1", "auth_token": "tok"}
    ]
    entries = [
        json.loads(line)
        for line in (tmp_path / "run" / "verdicts.jsonl").read_text().splitlines()
    ]
    assert entries == [
        {
            "task_id": "t-1",
            "image": "img:1",
            "verdict": "usable",
            "exit_code": 0,
            "error_category": None,
            "cached": False,
            "reward": None,
            "note": None,
        }
    ]


def test_main_requires_auth_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(_profile().model_dump_json())
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(json.dumps(_record()) + "\n")

    monkeypatch.delenv("PYROMIND_AUTH_TOKEN", raising=False)
    exit_code = runner.main(
        [
            "--profile",
            str(profile_path),
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(tmp_path / "run"),
            "--env",
            "pre",
            "--cluster",
            "us-west-1",
        ],
        client_factory=lambda **kwargs: _mock_client(),
    )
    assert exit_code == 2


# ---------------------------------------------------------------------------
# install_pi / run_pi / reward_file
# ---------------------------------------------------------------------------

_PI_SECRETS = {
    "LLM_BASE_URL": "https://gw.example/v1",
    "LLM_AUTH_TOKEN": "sk-secret",
    "LLM_MODEL": "openai/deepseek-v4-flash-0731",
}


def _pi_profile(**overrides: Any) -> ProcessingProfile:
    payload: dict[str, Any] = {
        "name": "pi-profile",
        "steps": [
            {"name": "create_sandbox", "params": {"image": "{image}"}},
            {
                "name": "install_pi",
                "params": {
                    "env": {
                        "LLM_BASE_URL": "{secret:LLM_BASE_URL}",
                        "LLM_AUTH_TOKEN": "{secret:LLM_AUTH_TOKEN}",
                        "LLM_MODEL": "{secret:LLM_MODEL}",
                    },
                },
            },
            {
                "name": "run_pi",
                "params": {
                    "prompt_field": "prompt",
                    "workdir": "{workdir}",
                    "env": {
                        "LLM_BASE_URL": "{secret:LLM_BASE_URL}",
                        "LLM_AUTH_TOKEN": "{secret:LLM_AUTH_TOKEN}",
                        "LLM_MODEL": "{secret:LLM_MODEL}",
                    },
                },
            },
            {
                "name": "write_file",
                "params": {"path": "/workspace/t.sh", "content_field": "test_sh"},
            },
            {"name": "exec", "params": {"command": "bash /workspace/t.sh"}},
            {"name": "delete_sandbox", "params": {}},
        ],
        "verdict": {"kind": "reward_file", "reward_path": "/logs/verifier/reward.txt"},
    }
    payload.update(overrides)
    return ProcessingProfile.model_validate(payload)


def test_tmax_profile_declares_llm_env_triple() -> None:
    """The shipped tmax profile must declare the full LLM env triple.

    Without the LLM_MODEL placeholder install_pi fails loudly instead of
    letting pi run against an unintended default model.
    """
    profile_path = _RUNNER_PATH.parent / "profiles" / "tmax-validation.json"
    profile = ProcessingProfile.model_validate_json(
        profile_path.read_text(encoding="utf-8")
    )
    for step_name in ("install_pi", "run_pi"):
        step = next(s for s in profile.steps if s.name == step_name)
        env = step.params["env"]
        assert env["LLM_BASE_URL"] == "{secret:LLM_BASE_URL}"
        assert env["LLM_AUTH_TOKEN"] == "{secret:LLM_AUTH_TOKEN}"
        assert env["LLM_MODEL"] == "{secret:LLM_MODEL}"


def test_install_pi_registers_gateway_in_models_json() -> None:
    client = _mock_client()
    step = ProcessingStep.model_validate(
        {"name": "install_pi", "params": {"env": dict(_PI_SECRETS)}}
    )

    runner._install_pi(step, _record(), client, "sb-1")

    command = client.exec_command.call_args.args[1]
    # Node provisioning pinned to pi's engines floor, then the agent itself.
    assert f"node-v{runner.PI_NODE_VERSION}" in command
    assert "@earendil-works/pi-coding-agent" in command
    # models.json lands under the run user's HOME with a /v1-normalized base.
    assert 'cat > "$HOME/.pi/agent/models.json"' in command
    assert '"baseUrl": "https://gw.example/v1"' in command
    assert '"id": "openai/deepseek-v4-flash-0731"' in command
    assert '"apiKey": "sk-secret"' in command


def _pi_mock_client(
    install_code: int = 0,
    pi_code: int = 0,
    verifier_code: int = 0,
    reward_raw: bytes | None = b"0.85\n",
    verifier_output: str = "",
    trace_raw: bytes | None = None,
) -> MagicMock:
    client = _mock_client()

    def _exec(sandbox_id: str, command: str, **kwargs: Any) -> SimpleNamespace:
        if "npm install --prefix /opt/pi" in command:
            return SimpleNamespace(returncode=install_code, output="ok")
        if "nohup bash -c" in command:
            return SimpleNamespace(returncode=0, output="launched")
        if f"test -f {runner.PI_EXIT_FILE}" in command:
            return SimpleNamespace(returncode=0, output=str(pi_code))
        if "pkill" in command or "tail -c" in command:
            return SimpleNamespace(returncode=0, output="pi stderr tail")
        if "/workspace/t.sh" in command:
            return SimpleNamespace(returncode=verifier_code, output=verifier_output)
        return SimpleNamespace(returncode=1, output=f"unexpected: {command}")

    def _read(sandbox_id: str, path: str, **kwargs: Any) -> bytes:
        if reward_raw is not None and path == "/logs/verifier/reward.txt":
            return reward_raw
        if trace_raw is not None and path == "/workspace/.pi_trace.jsonl":
            return trace_raw
        raise RuntimeError(f"cannot read {path}")

    client.exec_command.side_effect = _exec
    client.read_file.side_effect = _read
    return client


def test_validate_record_pi_chain_usable_with_reward() -> None:
    client = _pi_mock_client()
    entry = runner.validate_record(_pi_profile(), _record(), client, _PI_SECRETS)

    assert entry.verdict == "usable"
    assert entry.reward == 0.85
    assert entry.exit_code == 0
    # The prompt reaches the sandbox as a file, not as a shell argument.
    writes = {c.args[1]: c.args[2] for c in client.write_file.call_args_list}
    assert writes[runner.PI_TASK_FILE] == b"solve the task"
    script = writes[runner.PI_RUN_SCRIPT].decode()
    assert "cd /workspace" in script
    assert "export HOME=/workspace;" in script
    assert "--mode json --provider mygw" in script
    assert "--model openai/deepseek-v4-flash-0731" in script
    # Credentials live in models.json only; the launch script never sees them.
    assert "sk-secret" not in script
    launch_calls = [
        c for c in client.exec_command.call_args_list if "nohup bash -c" in str(c.args)
    ]
    assert len(launch_calls) == 1
    launch = launch_calls[0].args[1]
    assert launch.startswith(f"chmod 644 {runner.PI_TASK_FILE}")
    assert f"rm -f {runner.PI_EXIT_FILE}" in launch
    assert f"echo $? > {runner.PI_EXIT_FILE}" in launch
    assert launch.endswith("& echo launched")


def test_validate_record_pi_run_as_user_su_wraps_when_root() -> None:
    payload = json.loads(_pi_profile().model_dump_json())
    payload["steps"][2]["params"]["run_as_user"] = "user"
    profile = ProcessingProfile.model_validate(payload)
    client = _pi_mock_client()

    entry = runner.validate_record(profile, _record(), client, _PI_SECRETS)

    assert entry.verdict == "usable"
    launch_calls = [
        c for c in client.exec_command.call_args_list if "nohup bash -c" in str(c.args)
    ]
    command = launch_calls[0].args[1]
    assert command.startswith(f"chmod 644 {runner.PI_TASK_FILE}")
    assert "su -s /bin/bash user -c" in command
    assert "bash /workspace/.pi_run.sh" in command
    assert f"echo $? > {runner.PI_EXIT_FILE}" in command


def test_validate_record_pi_secrets_never_reach_verdicts(tmp_path: Path) -> None:
    client = _pi_mock_client()
    runner.run_batch(_pi_profile(), [_record()], tmp_path, client, secrets=_PI_SECRETS)

    verdicts_text = (tmp_path / "verdicts.jsonl").read_text()
    assert "sk-secret" not in verdicts_text
    assert "gw.example" not in verdicts_text


def test_validate_record_pi_install_failure() -> None:
    client = _pi_mock_client(install_code=1)
    entry = runner.validate_record(_pi_profile(), _record(), client, _PI_SECRETS)

    assert entry.verdict == "error"
    assert entry.error_category == runner.CATEGORY_PI_INSTALL_FAILED
    client.delete.assert_called_once_with("sb-1")


def test_validate_record_pi_run_failure() -> None:
    client = _pi_mock_client(pi_code=2)
    entry = runner.validate_record(_pi_profile(), _record(), client, _PI_SECRETS)

    assert entry.verdict == "error"
    assert entry.error_category == runner.CATEGORY_PI_RUN_FAILED


def test_validate_record_reward_file_missing_falls_back_to_exit_code() -> None:
    client = _pi_mock_client(reward_raw=None)
    entry = runner.validate_record(_pi_profile(), _record(), client, _PI_SECRETS)

    assert entry.verdict == "usable"
    assert entry.reward is None


def test_validate_record_missing_secret_is_error() -> None:
    client = _pi_mock_client()
    entry = runner.validate_record(_pi_profile(), _record(), client, {})

    assert entry.verdict == "error"
    assert entry.error_category == runner.CATEGORY_EXEC_FAILED


def test_run_pi_exports_trace_to_trace_dir(tmp_path: Path) -> None:
    payload = json.loads(_pi_profile().model_dump_json())
    payload["steps"][2]["params"]["export_trace"] = True
    profile = ProcessingProfile.model_validate(payload)
    trace = b'{"type":"agent_settled"}\n'
    client = _pi_mock_client(trace_raw=trace)

    entry = runner.validate_record(
        profile, _record(), client, _PI_SECRETS, trace_dir=tmp_path / "traces"
    )

    assert entry.verdict == "usable"
    exported = tmp_path / "traces" / "t-1.pi_trace.jsonl"
    assert exported.read_bytes() == trace


def test_run_pi_timeout_kills_and_fails() -> None:
    payload = json.loads(_pi_profile().model_dump_json())
    payload["steps"][2]["params"]["timeout"] = 0
    profile = ProcessingProfile.model_validate(payload)
    client = _pi_mock_client()
    original_exec = client.exec_command.side_effect

    def _stalled_exec(sandbox_id: str, command: str, **kwargs: Any) -> SimpleNamespace:
        if f"test -f {runner.PI_EXIT_FILE}" in command:
            return SimpleNamespace(returncode=0, output=runner.RUNNING_MARK)
        result = original_exec(sandbox_id, command, **kwargs)
        assert result is not None
        return result

    client.exec_command.side_effect = _stalled_exec

    entry = runner.validate_record(profile, _record(), client, _PI_SECRETS)

    assert entry.verdict == "error"
    assert entry.error_category == runner.CATEGORY_PI_RUN_FAILED
    kill_calls = [
        c for c in client.exec_command.call_args_list if "pkill" in str(c.args)
    ]
    assert len(kill_calls) == 1


def test_validate_record_verifier_env_missing_bucket() -> None:
    client = _pi_mock_client(
        verifier_code=1,
        reward_raw=None,
        verifier_output="/app/oracle_parser: No such file or directory",
    )
    entry = runner.validate_record(_pi_profile(), _record(), client, _PI_SECRETS)

    assert entry.verdict == "error"
    assert entry.error_category == runner.CATEGORY_VERIFIER_ENV_MISSING
    assert entry.note is not None and "image may lack artifacts" in entry.note


def test_validate_record_verifier_failure_without_env_marker_stays_generic() -> None:
    client = _pi_mock_client(
        verifier_code=1, reward_raw=None, verifier_output="assertion failed"
    )
    entry = runner.validate_record(_pi_profile(), _record(), client, _PI_SECRETS)

    assert entry.verdict == "error"
    assert entry.error_category == runner.CATEGORY_VERIFIER_FAILED
    assert entry.note is None
