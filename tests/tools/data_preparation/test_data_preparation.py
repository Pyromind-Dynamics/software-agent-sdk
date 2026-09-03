import ast
import importlib.util
import json
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any, cast

import pyarrow.parquet as pq
import pytest
from PIL import Image
from pydantic import SecretStr

from openhands.sdk.conversation.secret_registry import SecretRegistry
from openhands.sdk.llm import LLM, FailoverRouter
from openhands.sdk.workspace.workspace import LocalWorkspace
from openhands.tools.data_preparation.definition import (
    DEFAULT_SAMPLE_LIMIT,
    DF_SAMPLE_LIMIT_ENV,
    RUNTIME_FILENAMES,
    DfConvertAction,
    DfConvertExecutor,
    DfConvertObservation,
    DfRunPipelineAction,
    DfRunPipelineExecutor,
    DfRunPipelineObservation,
    _sample_limit,
    _truncate_sample_input,
)
from openhands.tools.data_preparation.runner import (
    ProcessLocalSampleExecutor,
    _concrete_llm,
    build_dataflow_env,
    openai_compatible_model_name,
    preflight_dataflow_llm,
    run_dataflow_python,
    runtime_public_names,
    summarize_dataflow_env,
    validate_managed_image_pipeline,
)
from openhands.tools.utils.dataflow_config import (
    DEFAULT_DATAFLOW_API_BASE_URL,
    DEFAULT_DATAFLOW_MODEL_NAME,
)


def test_process_local_sample_executor_interrupts_process_group(tmp_path: Path) -> None:
    executor = ProcessLocalSampleExecutor()
    result: list[tuple[int, str, str]] = []
    thread = threading.Thread(
        target=lambda: result.append(
            executor.run(
                sys.executable,
                ["-c", "import time; time.sleep(60)"],
                cwd=str(tmp_path),
                env_extra={},
                timeout=120,
            )
        )
    )
    thread.start()
    deadline = time.monotonic() + 5
    while executor._process is None and time.monotonic() < deadline:
        time.sleep(0.01)
    executor.interrupt()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert result and result[0][0] != 0


def _fake_conversation(
    tmp_path: Path,
    *,
    secret_registry: SecretRegistry | None = None,
    agent_state: dict[str, Any] | None = None,
):
    return type(
        "FakeConversation",
        (),
        {
            "workspace": LocalWorkspace(working_dir=str(tmp_path)),
            "state": type(
                "FakeState",
                (),
                {
                    "secret_registry": secret_registry or SecretRegistry(),
                    "agent_state": agent_state or {},
                },
            )(),
        },
    )()


def test_openai_compatible_model_name() -> None:
    assert openai_compatible_model_name("gpt-4o-mini") == "gpt-4o-mini"
    assert openai_compatible_model_name("openai/gpt-4o") == "gpt-4o"
    assert (
        openai_compatible_model_name("openrouter/anthropic/claude-3-5-sonnet")
        == "anthropic/claude-3-5-sonnet"
    )


def _conversation_with_llm(base_url: str | None = "https://example.com/v1/") -> Any:
    llm = type(
        "FakeLlm",
        (),
        {
            "api_key": SecretStr("secret"),
            "base_url": base_url,
            "model": "openai/vision-model",
        },
    )()
    return type(
        "FakeConversation",
        (),
        {
            "state": type(
                "FakeState",
                (),
                {"agent": type("FakeAgent", (), {"llm": llm})()},
            )()
        },
    )()


def test_build_dataflow_env_vision_falls_back_to_defaults(
    monkeypatch,
) -> None:
    for name in (
        "DF_API_KEY",
        "DF_API_URL",
        "DF_API_BASE_URL",
        "DF_MODEL_NAME",
        "LLM_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)

    env = build_dataflow_env(_conversation_with_llm(base_url=None))

    assert env == {
        "DF_API_KEY": "secret",
        "DF_API_URL": f"{DEFAULT_DATAFLOW_API_BASE_URL}/chat/completions",
        "DF_API_BASE_URL": DEFAULT_DATAFLOW_API_BASE_URL,
        "DF_MODEL_NAME": DEFAULT_DATAFLOW_MODEL_NAME,
    }


def test_build_dataflow_env_prefers_global_vision_model(monkeypatch) -> None:
    monkeypatch.setenv("DF_API_KEY", "openrouter-secret")
    monkeypatch.setenv("DF_API_BASE_URL", "https://openrouter.ai/api/v1/")
    monkeypatch.setenv(
        "DF_API_URL",
        "https://openrouter.ai/api/v1/chat/completions/",
    )
    monkeypatch.setenv("DF_MODEL_NAME", "google/gemma-4-31b-it")

    env = build_dataflow_env(_conversation_with_llm())

    assert env == {
        "DF_API_KEY": "openrouter-secret",
        "DF_API_URL": "https://openrouter.ai/api/v1/chat/completions",
        "DF_API_BASE_URL": "https://openrouter.ai/api/v1",
        "DF_MODEL_NAME": "google/gemma-4-31b-it",
    }
    summary = summarize_dataflow_env(env)
    assert "google/gemma-4-31b-it" in summary
    assert "openrouter-secret" not in summary


def test_build_dataflow_env_text_profile_uses_conversation_model(
    monkeypatch,
) -> None:
    monkeypatch.setenv("DF_API_KEY", "vision-secret")
    monkeypatch.setenv("DF_API_BASE_URL", "https://vision.example/v1")
    monkeypatch.setenv(
        "DF_API_URL",
        "https://vision.example/v1/chat/completions",
    )
    monkeypatch.setenv("DF_MODEL_NAME", "gemma")

    env = build_dataflow_env(_conversation_with_llm(), "text")

    assert env["DF_MODEL_NAME"] == "vision-model"
    assert env["DF_API_BASE_URL"] == "https://example.com/v1"
    assert env["DF_API_KEY"] == "secret"


def test_build_dataflow_env_derives_missing_url_pair(monkeypatch) -> None:
    monkeypatch.setenv("DF_API_KEY", "secret")
    monkeypatch.setenv(
        "DF_API_URL",
        "https://openrouter.ai/api/v1/chat/completions",
    )
    monkeypatch.delenv("DF_API_BASE_URL", raising=False)
    monkeypatch.setenv("DF_MODEL_NAME", "vision-model")

    env = build_dataflow_env(_conversation_with_llm())

    assert env["DF_API_BASE_URL"] == "https://openrouter.ai/api/v1"


def test_build_dataflow_env_rejects_mismatched_urls(monkeypatch) -> None:
    monkeypatch.setenv("DF_API_KEY", "secret")
    monkeypatch.setenv("DF_API_BASE_URL", "https://one.example/v1")
    monkeypatch.setenv(
        "DF_API_URL",
        "https://two.example/v1/chat/completions",
    )
    monkeypatch.setenv("DF_MODEL_NAME", "vision-model")

    with pytest.raises(ValueError, match="do not describe the same endpoint"):
        build_dataflow_env(_conversation_with_llm())


def test_build_dataflow_env_vision_falls_back_to_llm_base_url(monkeypatch) -> None:
    monkeypatch.setenv("DF_API_KEY", "openrouter-secret")
    monkeypatch.setenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.delenv("DF_API_BASE_URL", raising=False)
    monkeypatch.delenv("DF_API_URL", raising=False)
    monkeypatch.delenv("DF_MODEL_NAME", raising=False)

    env = build_dataflow_env(_conversation_with_llm())

    assert env["DF_API_BASE_URL"] == "https://openrouter.ai/api/v1"
    assert env["DF_API_URL"] == "https://openrouter.ai/api/v1/chat/completions"
    assert env["DF_MODEL_NAME"] == DEFAULT_DATAFLOW_MODEL_NAME
    assert env["DF_API_KEY"] == "openrouter-secret"


def test_build_dataflow_env_vision_falls_back_to_conversation_llm(
    monkeypatch,
) -> None:
    monkeypatch.delenv("DF_API_KEY", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("DF_API_BASE_URL", raising=False)
    monkeypatch.delenv("DF_API_URL", raising=False)
    monkeypatch.delenv("DF_MODEL_NAME", raising=False)

    env = build_dataflow_env(_conversation_with_llm())

    assert env["DF_API_BASE_URL"] == "https://example.com/v1"
    assert env["DF_API_URL"] == "https://example.com/v1/chat/completions"
    assert env["DF_MODEL_NAME"] == DEFAULT_DATAFLOW_MODEL_NAME
    assert env["DF_API_KEY"] == "secret"


def test_run_dataflow_python_redacts_api_key(tmp_path: Path) -> None:
    api_key = "secret-that-must-not-leak"

    return_code, stdout, stderr = run_dataflow_python(
        sys.executable,
        [
            "-c",
            (
                "import os, sys; "
                "print(os.environ['DF_API_KEY']); "
                "print(os.environ['DF_API_KEY'], file=sys.stderr)"
            ),
        ],
        cwd=str(tmp_path),
        env_extra={"DF_API_KEY": api_key},
        timeout=60,
    )

    assert return_code == 0
    assert api_key not in stdout + stderr
    assert "<redacted>" in stdout
    assert "<redacted>" in stderr


def test_df_run_pipeline_validates_output_and_writes_local_report(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("DF_SKIP_PREFLIGHT", "1")
    pipeline_dir = tmp_path / "public_data" / "data-preparation"
    pipeline_dir.mkdir(parents=True)
    pipeline = pipeline_dir / "pipeline.py"
    pipeline.write_text(
        "\n".join(
            [
                "import json, sys",
                "from image_utils import IMAGE_UTILS_API_VERSION",
                "row = {",
                "  'id': 'text-1',",
                "  'system_prompt': 'system',",
                "  'user_prompt': 'question',",
                "  'gt': 'answer',",
                "}",
                "with open(sys.argv[2], 'w', encoding='utf-8') as f:",
                "    f.write(json.dumps(row) + '\\n')",
            ]
        ),
        encoding="utf-8",
    )
    input_path = pipeline_dir / "input.jsonl"
    input_path.write_text("{}\n", encoding="utf-8")
    scripts_dir = (
        Path(__file__).parents[3]
        / ".agents"
        / "skills"
        / "data-processing"
        / "scripts"
        / "preparation"
    )
    conversation = cast(Any, _fake_conversation(tmp_path))
    conversation.state.agent = _conversation_with_llm().state.agent

    observation = DfRunPipelineExecutor(runtime_dir=str(scripts_dir))(
        DfRunPipelineAction(
            pipeline_path="public_data/data-preparation/pipeline.py",
            args=[
                "public_data/data-preparation/input.jsonl",
                "public_data/data-preparation/processed.sample.jsonl",
            ],
            output_schema="text",
            model_profile="text",
        ),
        conversation,
    )

    assert not observation.is_error
    assert observation.exit_code == 0
    assert observation.report_path is not None
    report = json.loads(Path(observation.report_path).read_text())
    assert report["status"] == "succeeded"
    assert report["validation"]["status"] == "passed"
    assert report["total_records_output"] == 1
    assert (pipeline_dir / "processed.sample.jsonl").is_file()
    assert not (pipeline_dir / "public_data").exists()
    for filename in RUNTIME_FILENAMES:
        assert not (pipeline_dir / filename).exists()
        assert (
            pipeline_dir / ".processed.sample.state" / "runtime" / filename
        ).is_file()
    assert observation.output_path is not None
    assert observation.record_count == 1
    assert observation.sample_records == [
        {
            "id": "text-1",
            "system_prompt": "system",
            "user_prompt": "question",
            "gt": "answer",
        }
    ]


def test_df_run_pipeline_validates_dpo_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("DF_SKIP_PREFLIGHT", "1")
    pipeline_dir = tmp_path / "public_data" / "data-preparation"
    pipeline_dir.mkdir(parents=True)
    pipeline = pipeline_dir / "pipeline.py"
    pipeline.write_text(
        "\n".join(
            [
                "import json, sys",
                "row = {",
                "  'id': 'dpo-1',",
                "  'system_prompt': 'system',",
                "  'user_prompt': 'question',",
                "  'gt': 'chosen answer',",
                "  'rejected_answer': 'rejected answer',",
                "}",
                "with open(sys.argv[2], 'w', encoding='utf-8') as f:",
                "    f.write(json.dumps(row) + '\\n')",
            ]
        ),
        encoding="utf-8",
    )
    (pipeline_dir / "input.jsonl").write_text("{}\n", encoding="utf-8")
    scripts_dir = (
        Path(__file__).parents[3]
        / ".agents"
        / "skills"
        / "data-processing"
        / "scripts"
        / "preparation"
    )
    conversation = cast(Any, _fake_conversation(tmp_path))
    conversation.state.agent = _conversation_with_llm().state.agent

    observation = DfRunPipelineExecutor(runtime_dir=str(scripts_dir))(
        DfRunPipelineAction(
            pipeline_path="public_data/data-preparation/pipeline.py",
            args=[
                "public_data/data-preparation/input.jsonl",
                "public_data/data-preparation/processed.sample.jsonl",
            ],
            output_schema="dpo",
            model_profile="text",
        ),
        conversation,
    )

    assert not observation.is_error
    assert observation.exit_code == 0
    assert observation.report_path is not None
    report = json.loads(Path(observation.report_path).read_text())
    assert report["validation"] == {"status": "passed", "schema": "dpo", "rows": 1}
    assert observation.sample_records == [
        {
            "id": "dpo-1",
            "system_prompt": "system",
            "user_prompt": "question",
            "gt": "chosen answer",
            "rejected_answer": "rejected answer",
        }
    ]


def test_df_run_pipeline_exposes_structured_missing_input_error(
    tmp_path: Path,
) -> None:
    pipeline_dir = tmp_path / "public_data" / "data-preparation"
    pipeline_dir.mkdir(parents=True)
    (pipeline_dir / "pipeline.py").write_text("print('not executed')\n")
    conversation = cast(Any, _fake_conversation(tmp_path))

    observation = DfRunPipelineExecutor()(
        DfRunPipelineAction(
            pipeline_path="public_data/data-preparation/pipeline.py",
            args=[
                "datasets/storage/source.jsonl",
                "public_data/data-preparation/processed.sample.jsonl",
            ],
            output_schema="dpo",
        ),
        conversation,
    )

    assert observation.is_error
    assert observation.failure_stage == "input_resolution"
    assert observation.error_code == "workspace_input_not_found"
    assert observation.error_message is not None
    assert "Missing or unreadable workspace input" in observation.error_message
    llm_text = "\n".join(item.text for item in observation.to_llm_content)
    assert "failure_stage=input_resolution" in llm_text
    assert "error_code=workspace_input_not_found" in llm_text
    assert "datasets/storage/source.jsonl" in llm_text
    assert "stdout (tail)" not in llm_text


def test_df_run_pipeline_observation_keeps_from_text_diagnostic() -> None:
    observation = DfRunPipelineObservation.from_text(
        text="controlled diagnostic",
        is_error=True,
        exit_code=2,
        failure_stage="pipeline_resolution",
        error_code="workspace_pipeline_not_found",
        error_message="controlled diagnostic",
    )

    llm_text = "\n".join(item.text for item in observation.to_llm_content)
    assert "error_message=controlled diagnostic" in llm_text
    assert "error_code=workspace_pipeline_not_found" in llm_text


def test_df_run_pipeline_classifies_pipeline_execution_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("DF_SKIP_PREFLIGHT", "1")
    pipeline_dir = tmp_path / "public_data" / "data-preparation"
    pipeline_dir.mkdir(parents=True)
    (pipeline_dir / "pipeline.py").write_text("raise SystemExit(7)\n")
    (pipeline_dir / "input.jsonl").write_text("{}\n")
    scripts_dir = (
        Path(__file__).parents[3]
        / ".agents"
        / "skills"
        / "data-processing"
        / "scripts"
        / "preparation"
    )
    conversation = cast(Any, _fake_conversation(tmp_path))
    conversation.state.agent = _conversation_with_llm().state.agent

    observation = DfRunPipelineExecutor(runtime_dir=str(scripts_dir))(
        DfRunPipelineAction(
            pipeline_path="public_data/data-preparation/pipeline.py",
            args=[
                "public_data/data-preparation/input.jsonl",
                "public_data/data-preparation/processed.sample.jsonl",
            ],
            output_schema="dpo",
        ),
        conversation,
    )

    assert observation.is_error
    assert observation.failure_stage == "pipeline_execution"
    assert observation.error_code == "dataflow_pipeline_failed"
    assert observation.error_message == "DataFlow pipeline exited with code 7."
    assert observation.exit_code == 7


def test_df_run_pipeline_uses_reported_failure_details(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("DF_SKIP_PREFLIGHT", "1")
    pipeline_dir = tmp_path / "public_data" / "data-preparation"
    pipeline_dir.mkdir(parents=True)
    (pipeline_dir / "pipeline.py").write_text(
        "import json, os\n"
        "from pathlib import Path\n"
        "failure = {\n"
        "    'status': 'failed',\n"
        "    'failure': {\n"
        "        'stage': 'input_validation',\n"
        "        'error': 'missing configured reference field label',\n"
        "        'attempts': 0,\n"
        "    },\n"
        "}\n"
        "state_dir = Path(os.environ['DF_STATE_DIR'])\n"
        "(state_dir / 'failure.json').write_text(json.dumps(failure))\n"
        "raise SystemExit(7)\n"
    )
    (pipeline_dir / "input.jsonl").write_text("{}\n")
    scripts_dir = (
        Path(__file__).parents[3]
        / ".agents"
        / "skills"
        / "data-processing"
        / "scripts"
        / "preparation"
    )
    conversation = cast(Any, _fake_conversation(tmp_path))
    conversation.state.agent = _conversation_with_llm().state.agent

    observation = DfRunPipelineExecutor(runtime_dir=str(scripts_dir))(
        DfRunPipelineAction(
            pipeline_path="public_data/data-preparation/pipeline.py",
            args=[
                "public_data/data-preparation/input.jsonl",
                "public_data/data-preparation/processed.sample.jsonl",
            ],
            output_schema="dpo",
        ),
        conversation,
    )

    assert observation.is_error
    assert observation.failure_stage == "input_validation"
    assert observation.error_code == "dataflow_pipeline_failed"
    assert observation.error_message == "missing configured reference field label"
    assert observation.exit_code == 7
    llm_text = "\n".join(item.text for item in observation.to_llm_content)
    assert "failure_stage=input_validation" in llm_text
    assert "error_message=missing configured reference field label" in llm_text


def test_df_run_pipeline_classifies_schema_validation_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("DF_SKIP_PREFLIGHT", "1")
    pipeline_dir = tmp_path / "public_data" / "data-preparation"
    pipeline_dir.mkdir(parents=True)
    (pipeline_dir / "pipeline.py").write_text(
        "import json, sys\n"
        "with open(sys.argv[2], 'w', encoding='utf-8') as f:\n"
        "    f.write(json.dumps({'id': 'missing-dpo-fields'}) + '\\n')\n"
    )
    (pipeline_dir / "input.jsonl").write_text("{}\n")
    scripts_dir = (
        Path(__file__).parents[3]
        / ".agents"
        / "skills"
        / "data-processing"
        / "scripts"
        / "preparation"
    )
    conversation = cast(Any, _fake_conversation(tmp_path))
    conversation.state.agent = _conversation_with_llm().state.agent

    observation = DfRunPipelineExecutor(runtime_dir=str(scripts_dir))(
        DfRunPipelineAction(
            pipeline_path="public_data/data-preparation/pipeline.py",
            args=[
                "public_data/data-preparation/input.jsonl",
                "public_data/data-preparation/processed.sample.jsonl",
            ],
            output_schema="dpo",
        ),
        conversation,
    )

    assert observation.is_error
    assert observation.failure_stage == "schema_validation"
    assert observation.error_code == "dataflow_schema_validation_failed"
    assert "schema validation" in (observation.error_message or "")


def test_df_run_pipeline_rejects_handwritten_vision_transport(
    tmp_path: Path,
) -> None:
    pipeline_dir = tmp_path / "public_data" / "data-preparation"
    pipeline_dir.mkdir(parents=True)
    (pipeline_dir / "pipeline.py").write_text(
        "import base64\n"
        "from image_utils import ImagePipelineConfig, "
        "run_image_pipeline_from_cli\n"
    )
    (pipeline_dir / "input.jsonl").write_text("{}\n")
    scripts_dir = (
        Path(__file__).parents[3]
        / ".agents"
        / "skills"
        / "data-processing"
        / "scripts"
        / "preparation"
    )
    conversation = cast(Any, _fake_conversation(tmp_path))
    conversation.state.agent = _conversation_with_llm().state.agent

    observation = DfRunPipelineExecutor(runtime_dir=str(scripts_dir))(
        DfRunPipelineAction(
            pipeline_path="public_data/data-preparation/pipeline.py",
            args=[
                "public_data/data-preparation/input.jsonl",
                "public_data/data-preparation/processed.sample.jsonl",
            ],
            output_schema="vision",
            model_profile="vision",
        ),
        conversation,
    )

    assert observation.is_error
    assert "cannot import base64" in observation.text


def test_df_convert_messages_format(tmp_path: Path) -> None:
    # Prepare input
    input_file = tmp_path / "input.jsonl"
    input_file.write_text(
        json.dumps({"problem": "1+1=?", "reasoning": "1+1=2", "answer": "2"})
        + "\n"
        + json.dumps({"problem": "2+2=?", "answer": "4"})
        + "\n"
        + json.dumps({"no_field": "x"})
    )
    output_file = tmp_path / "messages.jsonl"

    # Run convert
    action = DfConvertAction(
        input_path=str(input_file),
        output_path=str(output_file),
        format="messages",
        text_field="problem",
        reasoning_field="reasoning",
        answer_field="answer",
        system_prompt="Solve the problem step by step.",
    )
    executor = DfConvertExecutor()
    conv = _fake_conversation(tmp_path)
    obs = executor(action, conv)

    # Check output
    assert isinstance(obs, DfConvertObservation)
    assert obs.converted == 2
    assert obs.skipped == 1
    assert output_file.exists()
    records = [json.loads(line) for line in output_file.read_text().splitlines()]
    assert len(records) == 2
    assert records[0]["messages"][0]["content"] == "Solve the problem step by step."
    assert records[0]["messages"][1]["content"] == "1+1=?"
    assert records[0]["messages"][2]["content"] == "1+1=2\n\n2"
    assert records[1]["messages"][0]["content"] == "Solve the problem step by step."
    assert records[1]["messages"][1]["content"] == "2+2=?"
    assert records[1]["messages"][2]["content"] == "4"


def test_df_convert_preference_format(tmp_path: Path) -> None:
    # Prepare input
    input_file = tmp_path / "input.jsonl"
    input_file.write_text(
        json.dumps({"prompt": "1+1=?", "chosen": "2", "rejected": "3"})
        + "\n"
        + json.dumps({"prompt": "2+2=?", "chosen": "4"})
    )
    output_file = tmp_path / "preference.jsonl"

    # Run convert
    action = DfConvertAction(
        input_path=str(input_file),
        output_path=str(output_file),
        format="preference",
        text_field="prompt",
        chosen_field="chosen",
        rejected_field="rejected",
    )
    executor = DfConvertExecutor()
    conv = _fake_conversation(tmp_path)
    obs = executor(action, conv)

    assert isinstance(obs, DfConvertObservation)
    assert obs.converted == 1
    assert obs.skipped == 1
    assert obs.output_path == str(output_file)
    assert obs.columns == ["prompt", "chosen", "rejected"]
    assert output_file.exists()
    records = [json.loads(line) for line in output_file.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["prompt"] == "1+1=?"
    assert records[0]["chosen"] == "2"
    assert records[0]["rejected"] == "3"


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (4, 4), color).save(path)


def test_df_convert_trl_vision_sft_parquet(tmp_path: Path) -> None:
    _write_image(tmp_path / "front.jpg", (255, 0, 0))
    _write_image(tmp_path / "detail.png", (0, 255, 0))
    input_file = tmp_path / "processed.jsonl"
    input_file.write_text(
        json.dumps(
            {
                "training_prompt": "Compare the images.",
                "training_response": {"label": "ok", "reason": "They match."},
                "image_paths": ["front.jpg", "detail.png"],
                "image_labels": ["Front view", "Detail view"],
            }
        )
        + "\n"
        + json.dumps(
            {
                "training_prompt": "Missing response.",
                "image_paths": ["front.jpg"],
            }
        ),
        encoding="utf-8",
    )
    output_file = tmp_path / "train.parquet"
    action = DfConvertAction(
        input_path=str(input_file),
        output_path=str(output_file),
        format="trl_vision_sft",
    )

    obs = DfConvertExecutor()(action, _fake_conversation(tmp_path))

    assert not obs.is_error
    assert obs.converted == 1
    assert obs.skipped == 1
    assert obs.output_path == str(output_file)
    assert obs.columns == ["messages", "images"]
    table = pq.read_table(output_file)
    assert b"huggingface" in (table.schema.metadata or {})
    row = table.to_pylist()[0]
    assert [image["path"] for image in row["images"]] == ["front.jpg", "detail.png"]
    assert all(image["bytes"] for image in row["images"])
    user = row["messages"][0]
    assert user["role"] == "user"
    assert user["content"] == [
        {"type": "text", "text": "Compare the images."},
        {"type": "text", "text": "Front view"},
        {"type": "image", "text": None},
        {"type": "text", "text": "Detail view"},
        {"type": "image", "text": None},
    ]
    assistant = row["messages"][1]
    assert assistant["content"][0]["text"] == ('{"label":"ok","reason":"They match."}')

    try:
        from datasets import load_dataset
    except ImportError:
        return
    loaded: Any = load_dataset(
        "parquet",
        data_files=str(output_file),
        split="train",
        cache_dir=str(tmp_path / "hf-cache"),
    )
    decoded: Any = loaded[0]
    assert len(decoded["images"]) == 2
    assert all(isinstance(image, Image.Image) for image in decoded["images"])


def test_df_convert_trl_vision_rejects_mismatched_labels(tmp_path: Path) -> None:
    _write_image(tmp_path / "image.jpg", (0, 0, 255))
    input_file = tmp_path / "processed.jsonl"
    input_file.write_text(
        json.dumps(
            {
                "training_prompt": "Describe.",
                "training_response": "Done.",
                "image_paths": ["image.jpg"],
                "image_labels": ["one", "extra"],
            }
        ),
        encoding="utf-8",
    )
    output_file = tmp_path / "train.parquet"
    action = DfConvertAction(
        input_path=str(input_file),
        output_path=str(output_file),
        format="trl_vision_sft",
    )

    obs = DfConvertExecutor()(action, _fake_conversation(tmp_path))

    assert not obs.is_error
    assert obs.converted == 0
    assert obs.skipped == 1
    assert pq.read_table(output_file).num_rows == 0


def test_df_convert_trl_vision_rejects_unsafe_or_invalid_image(
    tmp_path: Path,
) -> None:
    outside = tmp_path.parent / "outside.jpg"
    _write_image(outside, (0, 0, 0))
    input_file = tmp_path / "processed.jsonl"
    input_file.write_text(
        json.dumps(
            {
                "training_prompt": "Describe.",
                "training_response": "Done.",
                "image_paths": [str(outside)],
            }
        ),
        encoding="utf-8",
    )
    output_file = tmp_path / "train.parquet"
    action = DfConvertAction(
        input_path=str(input_file),
        output_path=str(output_file),
        format="trl_vision_sft",
    )

    obs = DfConvertExecutor()(action, _fake_conversation(tmp_path))

    assert obs.is_error
    assert "outside the workspace" in obs.text

    corrupt = tmp_path / "corrupt.jpg"
    corrupt.write_bytes(b"not an image")
    input_file.write_text(
        json.dumps(
            {
                "training_prompt": "Describe.",
                "training_response": "Done.",
                "image_paths": ["corrupt.jpg"],
            }
        ),
        encoding="utf-8",
    )
    obs = DfConvertExecutor()(action, _fake_conversation(tmp_path))
    assert obs.is_error


def test_df_convert_vision_sft_flat_preserves_ordered_paths(tmp_path: Path) -> None:
    _write_image(tmp_path / "front.jpg", (255, 0, 0))
    _write_image(tmp_path / "detail.png", (0, 255, 0))
    input_file = tmp_path / "processed.jsonl"
    input_file.write_text(
        json.dumps(
            {
                "sample_id": "sample-001",
                "training_system_prompt": "Return the requested judgment.",
                "training_prompt": "Compare the images.",
                "training_response": (
                    "<think>The details agree.</think>\n\n"
                    '<answer>{"status":"ok"}</answer>'
                ),
                "image_paths": ["front.jpg", "detail.png"],
            }
        ),
        encoding="utf-8",
    )
    output_file = tmp_path / "train.parquet"
    action = DfConvertAction(
        input_path=str(input_file),
        output_path=str(output_file),
        format="vision_sft_flat",
        id_field="sample_id",
        system_prompt_field="training_system_prompt",
        prompt_field="training_prompt",
        response_field="training_response",
        images_field="image_paths",
    )

    obs = DfConvertExecutor()(action, _fake_conversation(tmp_path))

    assert not obs.is_error
    assert obs.converted == 1
    assert obs.skipped == 0
    assert obs.columns == [
        "id",
        "image_path",
        "images",
        "system_prompt",
        "user_prompt",
        "gt",
    ]
    assert obs.column_schema == {
        "id": "string",
        "image_path": "string",
        "images": "list<string>",
        "system_prompt": "string",
        "user_prompt": "string",
        "gt": "string",
    }
    assert obs.image_path_mode == "ordered paths (not embedded)"
    table = pq.read_table(output_file)
    assert table.column_names == obs.columns
    assert (
        table.schema.field("images").type.value_type
        == table.schema.field("image_path").type
    )
    assert table.to_pylist() == [
        {
            "id": "sample-001",
            "image_path": "front.jpg",
            "images": ["front.jpg", "detail.png"],
            "system_prompt": "Return the requested judgment.",
            "user_prompt": "Compare the images.",
            "gt": (
                '<think>The details agree.</think>\n\n<answer>{"status":"ok"}</answer>'
            ),
        }
    ]


@pytest.mark.parametrize(
    ("records", "message"),
    [
        (
            [
                {
                    "sample_id": "same",
                    "training_system_prompt": "system",
                    "training_prompt": "prompt",
                    "training_response": "<think>why</think>\n\n<answer>A</answer>",
                    "image_paths": ["image.jpg"],
                },
                {
                    "sample_id": "same",
                    "training_system_prompt": "system",
                    "training_prompt": "prompt",
                    "training_response": "<think>why</think>\n\n<answer>B</answer>",
                    "image_paths": ["image.jpg"],
                },
            ],
            "duplicate sample_id",
        ),
        (
            [
                {
                    "sample_id": "sample",
                    "training_system_prompt": "system",
                    "training_prompt": "prompt",
                    "training_response": "<answer>A</answer><think>why</think>",
                    "image_paths": ["image.jpg"],
                }
            ],
            "invalid tag order",
        ),
    ],
)
def test_df_convert_vision_sft_flat_rejects_invalid_records(
    tmp_path: Path,
    records: list[dict[str, Any]],
    message: str,
) -> None:
    _write_image(tmp_path / "image.jpg", (0, 0, 255))
    input_file = tmp_path / "processed.jsonl"
    input_file.write_text(
        "\n".join(json.dumps(record) for record in records),
        encoding="utf-8",
    )
    action = DfConvertAction(
        input_path=str(input_file),
        output_path=str(tmp_path / "train.parquet"),
        format="vision_sft_flat",
    )

    obs = DfConvertExecutor()(action, _fake_conversation(tmp_path))

    assert obs.is_error
    assert message in obs.text


def _load_skill_reference(name: str, *, stub_dataflow: bool = False):
    root = Path(__file__).parents[3]
    path = (
        root
        / ".agents"
        / "skills"
        / "data-processing"
        / "references"
        / "paradigms"
        / "llm-pipeline"
        / name
    )
    previous_dataflow = sys.modules.get("dataflow")
    if stub_dataflow:
        dataflow = types.ModuleType("dataflow")
        dataflow.__file__ = str(root / "fake-dataflow" / "__init__.py")
        sys.modules["dataflow"] = dataflow
    try:
        module_name = f"test_data_preparation_{path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if stub_dataflow:
            if previous_dataflow is None:
                sys.modules.pop("dataflow", None)
            else:
                sys.modules["dataflow"] = previous_dataflow


def _load_image_utils() -> Any:
    root = Path(__file__).parents[3]
    scripts_dir = (
        root / ".agents" / "skills" / "data-processing" / "scripts" / "preparation"
    )
    path = scripts_dir / "image_utils.py"
    module_name = "test_data_preparation_image_utils"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    # Mirror the platform runtime, where image_utils.py and preparation_runtime.py
    # share a PYTHONPATH entry, so image_utils can import the shared helpers.
    sys.path.insert(0, str(scripts_dir))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
        sys.path.remove(str(scripts_dir))
    return module


def test_data_preparation_runtime_is_python_310_compatible() -> None:
    scripts_dir = (
        Path(__file__).parents[3]
        / ".agents"
        / "skills"
        / "data-processing"
        / "scripts"
        / "preparation"
    )
    for path in sorted(scripts_dir.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(
            source,
            filename=str(path),
            feature_version=(3, 10),
        )
        direct_datetime_utc = {
            alias.name
            for node in tree.body
            if isinstance(node, ast.ImportFrom) and node.module == "datetime"
            for alias in node.names
            if alias.name == "UTC"
        }
        assert not direct_datetime_utc, (
            f"{path.name} imports datetime.UTC without a Python 3.10 fallback"
        )
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "typing":
                imported_names = {alias.name for alias in node.names}
                unsupported_typing = imported_names & {"Self", "TypeAliasType"}
                assert not unsupported_typing, (
                    f"{path.name} imports Python 3.11+ typing API(s): "
                    f"{sorted(unsupported_typing)}"
                )
            if isinstance(node, ast.Attribute):
                assert not (
                    isinstance(node.value, ast.Name)
                    and node.value.id == "datetime"
                    and node.attr == "UTC"
                ), f"{path.name} uses datetime.UTC, which requires Python 3.11"


def test_image_utils_multi_image_prompt_retry_and_output_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_utils = _load_image_utils()
    for name, color in (
        ("front.jpg", (1, 2, 3)),
        ("back.jpg", (4, 5, 6)),
        ("single.jpg", (7, 8, 9)),
    ):
        _write_image(tmp_path / name, color)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "first",
                        "images": ["front.jpg", "back.jpg"],
                        "user_prompt": "Compare both.",
                    }
                ),
                json.dumps(
                    {
                        "id": "second",
                        "images": ["single.jpg"],
                        "user_prompt": "Inspect one.",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    class FakeServing:
        def __init__(self) -> None:
            self.request_sizes: list[int] = []

        def generate_from_input_multi_images(
            self,
            image_paths,
            image_labels,
            *,
            user_prompts,
            **kwargs,
        ):
            del kwargs
            self.request_sizes.append(len(image_paths))
            if len(self.request_sizes) == 1:
                assert image_labels[0] == ["Image 1", "Image 2"]
                assert "Compare both." in user_prompts[0]
                return [
                    '{"reasoning":"first","answer":"A"}',
                    "not-json",
                ]
            assert len(image_paths) == 1
            return ['{"reasoning":"second","answer":"B"}']

        def cleanup(self):
            return None

    state_dir = tmp_path / "state"
    monkeypatch.setenv("DF_STATE_DIR", str(state_dir))
    monkeypatch.setenv("DF_RESUME", "0")
    fake = FakeServing()
    monkeypatch.setattr(image_utils, "_create_vlm_serving", lambda config: fake)
    config = image_utils.ImagePipelineConfig(
        labeling_system_prompt="Label.",
        training_system_prompt="Train.",
        batch_size=2,
    )
    output = tmp_path / "processed.sample.jsonl"

    image_utils.run_image_pipeline(config, str(manifest), str(output))

    rows = [
        json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert fake.request_sizes == [2, 1]
    assert [item["role"] for item in rows[0]["messages"]] == [
        "system",
        "user",
        "assistant",
    ]
    assert rows[0]["messages"][1]["content"] == [
        {"type": "image_url", "value": "front.jpg"},
        {"type": "image_url", "value": "back.jpg"},
        {"type": "text", "value": "Compare both."},
    ]
    assert rows[1]["messages"][2]["content"][0]["value"] == (
        "<think>second</think>\n\n<answer>B</answer>"
    )
    assert set(rows[0]) == {"messages"}


def test_image_utils_accepts_json_fence_and_logs_invalid_response_preview(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_utils = _load_image_utils()
    _write_image(tmp_path / "one.jpg", (1, 2, 3))
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"id": "one", "images": ["one.jpg"], "user_prompt": "Inspect."})
        + "\n",
        encoding="utf-8",
    )

    class FakeServing:
        calls = 0

        def generate_from_input_multi_images(self, *args, **kwargs):
            del args
            self.calls += 1
            assert kwargs["json_schema"]["required"] == ["reasoning", "answer"]
            if self.calls == 1:
                return ["not-json sk-secret"]
            return ['```json\n{"reasoning":"ok","answer":"A"}\n```']

        def cleanup(self):
            return None

    state_dir = tmp_path / "state"
    monkeypatch.setenv("DF_STATE_DIR", str(state_dir))
    monkeypatch.setenv("DF_RESUME", "0")
    monkeypatch.setenv("DF_API_KEY", "sk-secret")
    monkeypatch.setattr(
        image_utils, "_create_vlm_serving", lambda config: FakeServing()
    )
    config = image_utils.ImagePipelineConfig(
        labeling_system_prompt="Label.",
        training_system_prompt="Train.",
        max_attempts=2,
    )

    output = tmp_path / "processed.jsonl"
    image_utils.run_image_pipeline(config, str(manifest), str(output))

    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["messages"][2]["content"][0]["value"].endswith("<answer>A</answer>")
    calls = [
        json.loads(line)
        for line in (state_dir / "llm_calls.jsonl").read_text().splitlines()
    ]
    assert calls[0]["status"] == "invalid_output"
    assert calls[0]["parse_mode"] == "raw_json"
    assert calls[0]["response_preview"]["head"] == "not-json <redacted>"
    assert "response_preview" not in calls[1]
    assert calls[1]["parse_mode"] == "markdown_json_fence"


def test_image_utils_reconciles_sidecar_human_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_utils = _load_image_utils()
    for name, label in (("keep", "skip"), ("correct", "true")):
        sample = tmp_path / name
        sample.mkdir()
        _write_image(sample / "image.jpg", (1, 2, 3))
        (sample / "meta.json").write_text(
            json.dumps({"label": label, "note": f"human-{name}"}),
            encoding="utf-8",
        )

    class FakeServing:
        def generate_from_input_multi_images(self, *args, **kwargs):
            del args
            schema = kwargs["json_schema"]
            assert "label_review" in schema["properties"]
            prompts = kwargs["user_prompts"]
            assert "Human label" in prompts[0]
            return [
                json.dumps(
                    {
                        "reasoning": "obvious mismatch",
                        "answer": "机台误报",
                        "label_review": {
                            "decision": "correct",
                            "correction_reason": "人工标签与可见形态直接矛盾",
                            "visual_evidence": ["检出图与参考图的目标区域形态一致"],
                        },
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "reasoning": "matches",
                        "answer": "机台误报",
                        "label_review": {
                            "decision": "keep",
                            "correction_reason": "",
                            "visual_evidence": [],
                        },
                    },
                    ensure_ascii=False,
                ),
            ]

        def cleanup(self):
            return None

    state_dir = tmp_path / "state"
    monkeypatch.setenv("DF_STATE_DIR", str(state_dir))
    monkeypatch.setenv("DF_RESUME", "0")
    monkeypatch.setattr(
        image_utils, "_create_vlm_serving", lambda config: FakeServing()
    )
    config = image_utils.ImagePipelineConfig(
        labeling_system_prompt="Label.",
        training_system_prompt="Train.",
        user_prompt_template="Inspect.",
        metadata_filename="meta.json",
        reference_label_path="metadata.label",
        reference_note_path="metadata.note",
        reference_label_map={"skip": "机台误报", "true": "真实缺陷"},
        allow_reference_correction=True,
        batch_size=2,
    )

    output = tmp_path / "processed.jsonl"
    image_utils.run_image_pipeline(config, str(tmp_path), str(output))

    calls = [
        json.loads(line)
        for line in (state_dir / "llm_calls.jsonl").read_text().splitlines()
    ]
    decisions = [call["label_reconciliation"]["decision"] for call in calls]
    assert decisions == ["correct", "keep"]
    corrected = calls[0]["label_reconciliation"]
    assert corrected["original_label"] == "真实缺陷"
    assert corrected["final_label"] == "机台误报"
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert all(set(row) == {"messages"} for row in rows)


def test_image_utils_dataflow_checkpoint_resume_without_duplicates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_utils = _load_image_utils()
    for name in ("first.jpg", "second.jpg"):
        _write_image(tmp_path / name, (1, 2, 3))
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(
            json.dumps(
                {
                    "id": name,
                    "images": [f"{name}.jpg"],
                    "user_prompt": f"Inspect {name}.",
                }
            )
            for name in ("first", "second")
        )
        + "\n",
        encoding="utf-8",
    )

    class FailingServing:
        calls = 0

        def generate_from_input_multi_images(self, *args, **kwargs):
            del args, kwargs
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("boundary failure")
            return ['{"reasoning":"ok","answer":"A"}']

        def cleanup(self):
            return None

    state_dir = tmp_path / "state"
    output = tmp_path / "processed.jsonl"
    config = image_utils.ImagePipelineConfig(
        labeling_system_prompt="Label.",
        training_system_prompt="Train.",
        batch_size=1,
        max_attempts=1,
    )
    monkeypatch.setenv("DF_STATE_DIR", str(state_dir))
    monkeypatch.setenv("DF_RESUME", "0")
    monkeypatch.setattr(
        image_utils,
        "_create_vlm_serving",
        lambda config: FailingServing(),
    )
    with pytest.raises(ValueError, match="boundary failure"):
        image_utils.run_image_pipeline(config, str(manifest), str(output))

    first_rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["messages"][1]["content"][-1]["value"] for row in first_rows] == [
        "Inspect first."
    ]
    assert (state_dir / "image_pipeline_last_success_step.txt").read_text() == "0,1\n"

    class SuccessfulServing:
        def generate_from_input_multi_images(self, *args, **kwargs):
            del args, kwargs
            return ['{"reasoning":"fixed","answer":"B"}']

        def cleanup(self):
            return None

    monkeypatch.setenv("DF_RESUME", "1")
    monkeypatch.setattr(
        image_utils,
        "_create_vlm_serving",
        lambda config: SuccessfulServing(),
    )
    image_utils.run_image_pipeline(config, str(manifest), str(output))
    resumed = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["messages"][1]["content"][-1]["value"] for row in resumed] == [
        "Inspect first.",
        "Inspect second.",
    ]


def test_image_utils_converts_webp_for_dataflow(
    tmp_path: Path,
) -> None:
    image_utils = _load_image_utils()
    source = tmp_path / "image.webp"
    Image.new("RGB", (2, 2), "red").save(source)
    converted = image_utils._prepare_vlm_image(source, tmp_path / "state")
    assert converted.suffix == ".png"
    with Image.open(converted) as image:
        assert image.format == "PNG"


def test_image_utils_discards_uncheckpointed_batch_part(tmp_path: Path) -> None:
    image_utils = _load_image_utils()
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{"id":"one"}\n')
    state_dir = tmp_path / "state"
    output = tmp_path / "processed.jsonl"
    storage = image_utils.ManagedStreamBatchedFileStorage(
        first_entry_file_name=str(manifest),
        state_dir=state_dir,
        output_path=output,
    )
    storage.parts_dir.mkdir(parents=True)
    (storage.parts_dir / "part-00000000.jsonl").write_text('{"id":"one"}\n')
    (storage.parts_dir / "part-00000001.jsonl").write_text('{"id":"uncommitted"}\n')
    storage.checkpoint_path.write_text("0,1\n")

    storage.materialize_output()

    assert output.read_text() == '{"id":"one"}\n'
    assert not (storage.parts_dir / "part-00000001.jsonl").exists()


def test_managed_image_pipeline_static_contract(tmp_path: Path) -> None:
    image_utils_path = (
        Path(__file__).parents[3]
        / ".agents"
        / "skills"
        / "data-processing"
        / "scripts"
        / "preparation"
        / "image_utils.py"
    )
    public_names = runtime_public_names(image_utils_path)
    valid = tmp_path / "valid.py"
    valid.write_text(
        "from image_utils import ImagePipelineConfig, "
        "run_image_pipeline_from_cli\n"
        "CONFIG = ImagePipelineConfig("
        "labeling_system_prompt='x', training_system_prompt='y')\n"
        "run_image_pipeline_from_cli(CONFIG)\n"
    )
    validate_managed_image_pipeline(valid, public_names)

    invalid = tmp_path / "invalid.py"
    invalid.write_text(
        "import base64\n"
        "from image_utils import ImagePipelineConfig, "
        "run_image_pipeline_from_cli\n"
    )
    with pytest.raises(ValueError, match="cannot import base64"):
        validate_managed_image_pipeline(invalid, public_names)


def test_avi_manifest_adapter_is_only_a_boundary_example(tmp_path: Path) -> None:
    adapter = _load_skill_reference("avi_manifest_adapter.py")
    sample = tmp_path / "1_B1"
    sample.mkdir()
    for name, color in zip(
        ("defect.jpg", "diff.jpg", "gt.jpg"),
        ((255, 0, 0), (0, 255, 0), (0, 0, 255)),
        strict=True,
    ):
        _write_image(sample / name, color)
    (sample / "meta.json").write_text(
        json.dumps(
            {
                "id": "1/B1",
                "label": "skip",
                "note": "same as reference",
                "part": "part-1",
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "manifest.jsonl"

    assert adapter.convert(str(sample), str(output)) == 1

    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["sample_id"] == "1/B1"
    assert record["image_labels"] == ["AOI检出图", "差分图", "GT参考图"]
    assert record["reference_annotations"] == {
        "label": "skip",
        "note": "same as reference",
    }
    assert record["metadata"]["part"] == "part-1"


class _FakePreflightResponse:
    def __init__(self, status_code: int, text: str, content_type: str) -> None:
        self.status_code = status_code
        self.text = text
        self.headers = {"content-type": content_type}


def test_preflight_passes_on_200(monkeypatch) -> None:
    monkeypatch.setattr(
        "openhands.tools.data_preparation.runner.httpx.post",
        lambda *args, **kwargs: _FakePreflightResponse(
            200, '{"choices": [{"message": {"content": "p"}}]}', "application/json"
        ),
    )

    preflight_dataflow_llm(
        {
            "DF_API_URL": "https://openrouter.ai/api/v1/chat/completions",
            "DF_MODEL_NAME": "google/gemma-4-31b-it",
        }
    )


def test_preflight_rejects_unknown_model(monkeypatch) -> None:
    monkeypatch.setattr(
        "openhands.tools.data_preparation.runner.httpx.post",
        lambda *args, **kwargs: _FakePreflightResponse(
            404,
            '{"error": {"message": "No endpoints found for router"}}',
            "application/json",
        ),
    )

    with pytest.raises(ValueError, match="rejected model 'router'"):
        preflight_dataflow_llm(
            {
                "DF_API_URL": "https://openrouter.ai/api/v1/chat/completions",
                "DF_MODEL_NAME": "router",
            }
        )


def test_preflight_detects_html_endpoint_hit(monkeypatch) -> None:
    monkeypatch.setattr(
        "openhands.tools.data_preparation.runner.httpx.post",
        lambda *args, **kwargs: _FakePreflightResponse(
            200, "<html><body>home</body></html>", "text/html; charset=utf-8"
        ),
    )

    with pytest.raises(ValueError, match="non-JSON"):
        preflight_dataflow_llm(
            {
                "DF_API_URL": "https://openrouter.ai/api/v1",
                "DF_MODEL_NAME": "google/gemma-4-31b-it",
            }
        )


def test_preflight_reports_auth_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        "openhands.tools.data_preparation.runner.httpx.post",
        lambda *args, **kwargs: _FakePreflightResponse(
            401, '{"error": "unauthorized"}', "application/json"
        ),
    )

    with pytest.raises(ValueError, match="auth failed"):
        preflight_dataflow_llm(
            {
                "DF_API_URL": "https://openrouter.ai/api/v1/chat/completions",
                "DF_MODEL_NAME": "google/gemma-4-31b-it",
            }
        )


def test_preflight_can_be_skipped(monkeypatch) -> None:
    monkeypatch.setenv("DF_SKIP_PREFLIGHT", "1")
    monkeypatch.setattr(
        "openhands.tools.data_preparation.runner.httpx.post",
        lambda *args, **kwargs: pytest.fail("preflight should be skipped"),
    )

    preflight_dataflow_llm(
        {
            "DF_API_URL": "https://openrouter.ai/api/v1/chat/completions",
            "DF_MODEL_NAME": "google/gemma-4-31b-it",
        }
    )


def test_preflight_reports_connection_failure(monkeypatch) -> None:
    import httpx as _httpx

    def _raise(*args, **kwargs):
        raise _httpx.ConnectError("unreachable")

    monkeypatch.setattr("openhands.tools.data_preparation.runner.httpx.post", _raise)

    with pytest.raises(ValueError, match="could not reach"):
        preflight_dataflow_llm(
            {
                "DF_API_URL": "https://openrouter.ai/api/v1/chat/completions",
                "DF_MODEL_NAME": "google/gemma-4-31b-it",
            }
        )


@pytest.mark.parametrize(
    "placeholder",
    ["router", "{{model}}", "{model}", "<model>", "${MODEL_NAME}", "x{{y}}"],
)
def test_build_dataflow_env_rejects_unsubstituted_placeholder_model(
    monkeypatch,
    placeholder: str,
) -> None:
    monkeypatch.setenv("DF_API_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("DF_MODEL_NAME", placeholder)

    with pytest.raises(ValueError, match="unsubstituted placeholder"):
        build_dataflow_env(_conversation_with_llm())


def test_build_dataflow_env_empty_model_falls_back_to_default(monkeypatch) -> None:
    for name in ("DF_MODEL_NAME", "DF_API_BASE_URL", "DF_API_URL"):
        monkeypatch.delenv(name, raising=False)

    env = build_dataflow_env(_conversation_with_llm(base_url="https://x.example"))

    assert env["DF_MODEL_NAME"] == DEFAULT_DATAFLOW_MODEL_NAME


def test_build_dataflow_env_text_profile_rejects_placeholder_llm_model(
    monkeypatch,
) -> None:
    """Text profile takes the model from the conversation LLM, so a routing
    placeholder there (e.g. ``router``) must be caught even when the process
    env names a real model."""
    for name in ("DF_API_KEY", "DF_MODEL_NAME", "DF_API_BASE_URL", "DF_API_URL"):
        monkeypatch.delenv(name, raising=False)
    conversation = _conversation_with_llm()
    conversation.state.agent.llm.model = "router"

    with pytest.raises(ValueError, match="unsubstituted placeholder"):
        build_dataflow_env(conversation, "text")


def test_build_dataflow_env_text_profile_uses_primary_router_provider(
    monkeypatch,
) -> None:
    """A RouterLLM (no real ``model``) must resolve the primary provider's
    model for DataFlow instead of failing on the ``router`` placeholder."""
    for name in ("DF_API_KEY", "DF_MODEL_NAME", "DF_API_BASE_URL", "DF_API_URL"):
        monkeypatch.delenv(name, raising=False)

    router = FailoverRouter(
        llms_for_routing={
            "deepseek": LLM(
                model="openai/deepseek-v4-flash-0731",
                base_url="http://208.64.254.187:8000/v1",
                api_key=SecretStr("first-key"),
            ),
            "openrouter": LLM(
                model="openai/deepseek-v4-flash-0731",
                base_url="https://openrouter.ai/api/v1",
                api_key=SecretStr("second-key"),
            ),
        },
    )
    conversation = _conversation_with_llm()
    conversation.state.agent.llm = router

    env = build_dataflow_env(conversation, "text")

    assert env["DF_MODEL_NAME"] == "deepseek-v4-flash-0731"
    assert env["DF_API_BASE_URL"] == "http://208.64.254.187:8000/v1"
    assert env["DF_API_KEY"] == "first-key"


def test_concrete_llm_delegates_to_router_primary() -> None:
    router = FailoverRouter(
        llms_for_routing={
            "deepseek": LLM(model="openai/deepseek-v4-flash-0731"),
            "openrouter": LLM(model="openai/deepseek-v4-flash-0731"),
        },
    )
    assert _concrete_llm(router).model == "openai/deepseek-v4-flash-0731"
    plain = LLM(model="openai/gpt-x")
    assert _concrete_llm(plain) is plain


def test_truncate_sample_input_csv_caps_rows(tmp_path: Path) -> None:
    input_path = tmp_path / "testdata_3.csv"
    input_path.write_text(
        "qid,question_text\n"
        + "\n".join(f"id{i},question {i}" for i in range(1, 8))  # 7 data rows
        + "\n",
        encoding="utf-8",
    )

    sample = _truncate_sample_input(input_path, 3)

    assert sample is not None
    content = sample.read_text(encoding="utf-8").splitlines()
    assert content[0] == "qid,question_text"  # header preserved
    assert len(content) == 1 + 3  # header + 3 data rows


def test_truncate_sample_input_jsonl_caps_rows(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    input_path.write_text("".join(f'{{"id": {i}}}\n' for i in range(5)))

    sample = _truncate_sample_input(input_path, 3)

    assert sample is not None
    assert len(sample.read_text(encoding="utf-8").splitlines()) == 3


def test_truncate_sample_input_skips_within_limit(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    input_path.write_text('{"id": 1}\n{"id": 2}\n')

    assert _truncate_sample_input(input_path, 3) is None


def test_truncate_sample_input_skips_directory(tmp_path: Path) -> None:
    directory = tmp_path / "images"
    directory.mkdir()

    assert _truncate_sample_input(directory, 3) is None


def test_sample_limit_env_and_default(monkeypatch) -> None:
    monkeypatch.delenv(DF_SAMPLE_LIMIT_ENV, raising=False)
    assert _sample_limit() == DEFAULT_SAMPLE_LIMIT
    monkeypatch.setenv(DF_SAMPLE_LIMIT_ENV, "10")
    assert _sample_limit() == 10
    monkeypatch.setenv(DF_SAMPLE_LIMIT_ENV, "garbage")
    assert _sample_limit() == DEFAULT_SAMPLE_LIMIT


def test_df_run_pipeline_caps_local_sample_rows(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A pipeline without ``--limit`` must still only see a 3-row local sample."""
    monkeypatch.setenv("DF_SKIP_PREFLIGHT", "1")
    monkeypatch.setenv("DF_SAMPLE_LIMIT", "3")
    pipeline_dir = tmp_path / "public_data" / "data-preparation"
    pipeline_dir.mkdir(parents=True)
    pipeline = pipeline_dir / "pipeline.py"
    # Copied input lines verbatim into text-schema records, so output rows ==
    # input rows the pipeline actually received.
    pipeline.write_text(
        "\n".join(
            [
                "import json, sys",
                "inp = sys.argv[1]",
                "out = sys.argv[2]",
                "rows = [l for l in open(inp, encoding='utf-8') if l.strip()]",
                "with open(out, 'w', encoding='utf-8') as f:",
                "    for i, l in enumerate(rows):",
                "        f.write(json.dumps({'id': f'text-{i}',",
                "            'system_prompt': 's', 'user_prompt': l.strip(),",
                "            'gt': 'a'}) + '\\n')",
            ]
        ),
        encoding="utf-8",
    )
    (pipeline_dir / "input.jsonl").write_text(
        "".join(f"q{i}\n" for i in range(5)), encoding="utf-8"
    )
    scripts_dir = (
        Path(__file__).parents[3]
        / ".agents"
        / "skills"
        / "data-processing"
        / "scripts"
        / "preparation"
    )
    conversation = cast(Any, _fake_conversation(tmp_path))
    conversation.state.agent = _conversation_with_llm().state.agent

    observation = DfRunPipelineExecutor(runtime_dir=str(scripts_dir))(
        DfRunPipelineAction(
            pipeline_path="public_data/data-preparation/pipeline.py",
            args=[
                "public_data/data-preparation/input.jsonl",
                "public_data/data-preparation/sample3.jsonl",
            ],
            output_schema="text",
            model_profile="text",
        ),
        conversation,
    )

    assert not observation.is_error, observation.text
    assert observation.record_count == 3  # capped at DF_SAMPLE_LIMIT, not 5


def _write_legacy_copy_pipeline(pipeline: Path) -> None:
    """Write a minimal legacy pipeline that copies input lines to the output."""
    pipeline.write_text(
        "\n".join(
            [
                "import sys",
                "with open(sys.argv[1], encoding='utf-8') as src:",
                "    lines = src.readlines()",
                "with open(sys.argv[2], 'w', encoding='utf-8') as dst:",
                "    dst.writelines(lines)",
            ]
        ),
        encoding="utf-8",
    )


def test_df_run_pipeline_legacy_accepts_workspace_relative_args(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Legacy runs (no output_schema) must accept workspace-relative args.

    The child process runs with cwd=pipeline.parent, so passing
    ``public_data/...`` used to double-prefix the path and fail with
    FileNotFoundError; args[0] is now resolved from the workspace root first.
    """
    monkeypatch.setenv("DF_SKIP_PREFLIGHT", "1")
    pipeline_dir = tmp_path / "public_data" / "data-preparation"
    pipeline_dir.mkdir(parents=True)
    _write_legacy_copy_pipeline(pipeline_dir / "pipeline.py")
    (pipeline_dir / "input.jsonl").write_text("{'a': 1}\n", encoding="utf-8")
    scripts_dir = (
        Path(__file__).parents[3]
        / ".agents"
        / "skills"
        / "data-processing"
        / "scripts"
        / "preparation"
    )
    conversation = cast(Any, _fake_conversation(tmp_path))
    conversation.state.agent = _conversation_with_llm().state.agent

    observation = DfRunPipelineExecutor(runtime_dir=str(scripts_dir))(
        DfRunPipelineAction(
            pipeline_path="public_data/data-preparation/pipeline.py",
            args=[
                "public_data/data-preparation/input.jsonl",
                "public_data/data-preparation/filtered.sample.jsonl",
            ],
            model_profile="text",
        ),
        conversation,
    )

    assert not observation.is_error, observation.text
    assert observation.exit_code == 0
    assert (pipeline_dir / "filtered.sample.jsonl").is_file()
    # Regression guard: the input must not be double-prefixed against the
    # pipeline directory.
    assert not (tmp_path / "public_data" / "public_data").exists()


def test_df_run_pipeline_legacy_keeps_pipeline_relative_args(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Legacy args relative to the pipeline directory keep working."""
    monkeypatch.setenv("DF_SKIP_PREFLIGHT", "1")
    pipeline_dir = tmp_path / "public_data" / "data-preparation"
    pipeline_dir.mkdir(parents=True)
    _write_legacy_copy_pipeline(pipeline_dir / "pipeline.py")
    (pipeline_dir / "input.jsonl").write_text("{'a': 1}\n", encoding="utf-8")
    scripts_dir = (
        Path(__file__).parents[3]
        / ".agents"
        / "skills"
        / "data-processing"
        / "scripts"
        / "preparation"
    )
    conversation = cast(Any, _fake_conversation(tmp_path))
    conversation.state.agent = _conversation_with_llm().state.agent

    observation = DfRunPipelineExecutor(runtime_dir=str(scripts_dir))(
        DfRunPipelineAction(
            pipeline_path="public_data/data-preparation/pipeline.py",
            args=["input.jsonl", "filtered.sample.jsonl"],
            model_profile="text",
        ),
        conversation,
    )

    assert not observation.is_error, observation.text
    assert observation.exit_code == 0
    assert (pipeline_dir / "filtered.sample.jsonl").is_file()
    assert not (tmp_path / "filtered.sample.jsonl").exists()
