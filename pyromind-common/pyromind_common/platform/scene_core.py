"""Pure business validation and graph builders shared by Tools and SDK scripts."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import shlex
import uuid
from collections.abc import Iterable, Sequence
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from pyromind_sdk.client.base import ENV_CLUSTER, parse_cluster_code

from pyromind_common.platform.paths import to_tool_storage_path
from pyromind_common.platform.scene_options import CleaningOptions, PipelineOptions


UTC = timezone(timedelta(0))
CLEANING_RUNTIME_FILENAMES = ("cleaning_utils.py", "validate_format.py")
ENV_DF_API_KEY = "DF_API_KEY"
ENV_DF_API_URL = "DF_API_URL"
ENV_DF_API_BASE_URL = "DF_API_BASE_URL"
ENV_DF_MODEL_NAME = "DF_MODEL_NAME"


def _pod_path(storage_path: str) -> str:
    return f"/target-workspace{storage_path}"


SUPPORTED_DATAFLOW_VERSION = "1.0.13"


DATAFLOW_REQUIREMENTS_FILENAME = "requirements-cpu.txt"


DATAFLOW_RUNTIME_FILENAMES = (
    "avi_pcb_runtime.py",
    "df_logging.py",
    "generate_report.py",
    "image_utils.py",
    "preparation_runtime.py",
    DATAFLOW_REQUIREMENTS_FILENAME,
    "source_fingerprint.py",
    "validate_prepared_data.py",
)


_FORBIDDEN_MANAGED_IMAGE_IMPORTS = {
    "base64",
    "dataflow",
    "httpx",
    "openai",
    "preparation_runtime",
    "requests",
    "urllib",
}


def _pip_install_command(
    *, pip_path: str, arguments: Sequence[str], cluster: str | None = None
) -> str:
    options = ["--timeout", "20", "--retries", "2", "--progress-bar", "off"]
    route = cluster if cluster is not None else os.environ.get(ENV_CLUSTER, "")
    if parse_cluster_code(route)[0] == "cn-east-1":
        options.extend(["--index-url", "https://mirrors.aliyun.com/pypi/simple/"])
    wrapper = (
        "import subprocess, sys\n"
        "try:\n"
        "    result = subprocess.run(sys.argv[1:], timeout=600)\n"
        "except subprocess.TimeoutExpired:\n"
        "    print('Dependency installation timed out after 600s', file=sys.stderr)\n"
        "    sys.exit(124)\n"
        "sys.exit(result.returncode)\n"
    )
    return " ".join(
        shlex.quote(part)
        for part in [
            "python3",
            "-c",
            wrapper,
            pip_path,
            "install",
            *arguments,
            *options,
        ]
    )


def dataflow_pip_install_commands(
    *, pip_path: str, requirements_path: str, cluster: str | None = None
) -> tuple[str, str]:
    return (
        _pip_install_command(
            pip_path=pip_path,
            arguments=["-r", requirements_path],
            cluster=cluster,
        ),
        _pip_install_command(
            pip_path=pip_path,
            arguments=["--no-deps", f"open-dataflow=={SUPPORTED_DATAFLOW_VERSION}"],
            cluster=cluster,
        ),
    )


def runtime_bundle_fingerprint(
    runtime_dir: Path,
    filenames: Iterable[str],
) -> str:
    """Hash runtime names and bytes in a stable order."""

    digest = hashlib.sha256()
    for filename in sorted(filenames):
        path = runtime_dir / filename
        if not path.is_file():
            raise ValueError(f"DataFlow runtime file is missing: {path}")
        digest.update(filename.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def runtime_public_names(path: Path) -> set[str]:
    """Read a literal ``__all__`` contract without importing the runtime."""

    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as exc:
        raise ValueError(f"Could not inspect image_utils public API: {exc}") from exc
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in targets
        ):
            continue
        if node.value is None:
            break
        try:
            value = ast.literal_eval(node.value)
        except (TypeError, ValueError) as exc:
            raise ValueError("image_utils __all__ must be a literal list") from exc
        if not isinstance(value, list) or not all(
            isinstance(name, str) for name in value
        ):
            raise ValueError("image_utils __all__ must contain only strings")
        return set(value)
    raise ValueError("image_utils does not define a public __all__ contract")


def validate_managed_image_pipeline(
    pipeline: Path,
    public_names: set[str],
) -> None:
    """Enforce the configuration-only managed image pipeline contract."""

    try:
        source = pipeline.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(pipeline))
        compile(tree, str(pipeline), "exec")
    except (OSError, SyntaxError) as exc:
        raise ValueError(f"Could not parse image pipeline: {exc}") from exc

    imported_image_utils: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.AsyncFor, ast.While, ast.comprehension)):
            raise ValueError(
                "Managed image pipelines must be configuration-only and cannot "
                "contain for/while loops."
            )
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if alias.name == "image_utils":
                    raise ValueError(
                        "Use explicit `from image_utils import ...` imports."
                    )
                if root in _FORBIDDEN_MANAGED_IMAGE_IMPORTS:
                    raise ValueError(f"Managed image pipelines cannot import {root}.")
        if not isinstance(node, ast.ImportFrom):
            continue
        module = node.module or ""
        root = module.split(".", 1)[0]
        if root in _FORBIDDEN_MANAGED_IMAGE_IMPORTS:
            raise ValueError(f"Managed image pipelines cannot import {root}.")
        if module != "image_utils":
            continue
        imported = {alias.name for alias in node.names}
        if "*" in imported:
            raise ValueError("image_utils wildcard imports are not allowed.")
        unknown = sorted(imported - public_names)
        if unknown:
            raise ValueError(
                "Managed image pipeline imports unsupported image_utils APIs: "
                + ", ".join(unknown)
            )
        imported_image_utils.update(imported)

    required = {"ImagePipelineConfig", "run_image_pipeline_from_cli"}
    missing = sorted(required - imported_image_utils)
    if missing:
        raise ValueError("Managed image pipeline must import: " + ", ".join(missing))


RUNTIME_FILENAMES = DATAFLOW_RUNTIME_FILENAMES


DATAFLOW_NODE_TYPE = "CustomCommandCPUNode"


_LEGACY_RUNTIME_PACKAGES = (
    "numpy==1.26.4",
    "pandas==2.3.3",
    "pyarrow==20.0.0",
    "Pillow==12.1.1",
    "opencv-python-headless==4.10.0.84",
    "matplotlib==3.9.4",
    "cloudpickle",
    "colorama",
    "colorlog",
    "datasketch",
    'fasttext; python_version >= "3.13"',
    'fasttext-wheel; python_version < "3.13"',
    "httpx",
    "huggingface-hub",
    "jsonschema",
    "math-verify",
    "nltk",
    "openai",
    "psutil",
    "pydantic",
    "PyYAML",
    "rapidfuzz",
    "redis",
    "requests",
    "rich",
    "tqdm",
    "word2number",
    "xxhash",
)


def _build_dataflow_command(
    *,
    input_path: str,
    output_dir: str,
    llm_env: dict[str, str],
    convert_format: str,  # noqa: ARG001
    frozen_script_name: str = "pipeline.py",
    resumed: bool = False,
    execution_revision: int = 1,
    runtime_dir_name: str = "",
    runtime_storage_dir: str | None = None,
    runtime_fingerprint: str | None = None,
    reuse_runtime: bool = False,
    image_utils_api_version: str | None = None,
    output_schema: str | None = None,
    reuse_assessment: dict[str, Any] | None = None,
    support_file_name: str | None = None,
    cluster: str | None = None,
) -> str:
    """Assemble the shell command executed inside the CustomCommandNode Pod."""
    pod_input = _pod_path(input_path)
    pod_output_dir = _pod_path(output_dir)
    if runtime_storage_dir:
        pod_runtime_dir = _pod_path(runtime_storage_dir)
    else:
        pod_runtime_dir = (
            f"{pod_output_dir}/{runtime_dir_name}"
            if runtime_dir_name
            else pod_output_dir
        )
    frozen_script = f"{pod_output_dir}/{frozen_script_name}"
    output_file = f"{pod_output_dir}/processed.jsonl"
    venv_python = "/tmp/df-venv/bin/python"
    setup_log = f"{pod_output_dir}/setup.log"
    pipeline_log = f"{pod_output_dir}/pipeline.log"
    validation_log = f"{pod_output_dir}/validation.log"
    report_log = f"{pod_output_dir}/report.log"
    runtime_files = (
        RUNTIME_FILENAMES
        if runtime_dir_name or runtime_storage_dir
        else tuple(name for name in RUNTIME_FILENAMES if name != "image_utils.py")
    )
    requirements = f"{pod_runtime_dir}/{DATAFLOW_REQUIREMENTS_FILENAME}"

    # Environment variables for LLM access
    env_parts = []
    for key in (
        ENV_DF_API_KEY,
        ENV_DF_API_URL,
        ENV_DF_API_BASE_URL,
        ENV_DF_MODEL_NAME,
    ):
        value = llm_env.get(key, "")
        if value:
            env_parts.append(f"{key}={shlex.quote(value)}")
    env_parts.append(f"DF_LOG_DIR={shlex.quote(pod_output_dir)}")
    env_parts.append(f"DF_STATE_DIR={shlex.quote(pod_output_dir)}")
    if output_schema is not None:
        env_parts.append(f"DF_OUTPUT_SCHEMA={shlex.quote(output_schema)}")
    env_parts.append(f"DF_RESUME={'1' if resumed else '0'}")
    env_parts.append(f"DF_EXECUTION_REVISION={execution_revision}")
    env_parts.append(f"PYTHONPATH={shlex.quote(pod_runtime_dir)}")
    if runtime_fingerprint:
        env_parts.append(f"DF_RUNTIME_FINGERPRINT={shlex.quote(runtime_fingerprint)}")
    if reuse_assessment is not None:
        env_parts.append(
            "DF_REUSE_ASSESSMENT_JSON="
            + shlex.quote(json.dumps(reuse_assessment, ensure_ascii=False))
        )
    env_prefix = " ".join(env_parts)

    # Pipeline arguments
    pipeline_args = [shlex.quote(pod_input), shlex.quote(output_file)]
    if support_file_name:
        pipeline_args.append(shlex.quote(f"{pod_output_dir}/{support_file_name}"))

    # Build the full command chain
    pip_path = "/tmp/df-venv/bin/pip"
    install_commands = dataflow_pip_install_commands(
        pip_path=pip_path, requirements_path=requirements, cluster=cluster
    )
    setup_output = f">> {shlex.quote(setup_log)} 2>&1"
    install = " && ".join(f"{command} {setup_output}" for command in install_commands)
    required_runtime_files = runtime_files
    if reuse_runtime:
        # Older runtime bundles were published without the dependency manifest.
        legacy_install = _pip_install_command(
            pip_path=pip_path,
            arguments=list(_LEGACY_RUNTIME_PACKAGES),
            cluster=cluster,
        )
        install = (
            f"if [ -f {shlex.quote(requirements)} ]; then {install}; else "
            f"{legacy_install} {setup_output} && "
            f"{install_commands[1]} {setup_output}; fi"
        )
        required_runtime_files = tuple(
            name for name in runtime_files if name != DATAFLOW_REQUIREMENTS_FILENAME
        )
    setup_steps = [
        f"mkdir -p {shlex.quote(pod_output_dir)}",
        f"test -f {shlex.quote(frozen_script)}",
        *[
            f"test -f {shlex.quote(f'{pod_runtime_dir}/{filename}')}"
            for filename in required_runtime_files
        ],
        f"python3 -m venv /tmp/df-venv > {shlex.quote(setup_log)} 2>&1",
        install,
    ]
    fingerprint_tool = f"{pod_runtime_dir}/source_fingerprint.py"
    setup_steps.append(
        "source_fingerprint_before=$("
        f"{venv_python} {shlex.quote(fingerprint_tool)} "
        f"{shlex.quote(pod_input)})"
    )
    if resumed:
        setup_steps.append(
            "{ resume_source_guard_rc=0; "
            f"{venv_python} {shlex.quote(fingerprint_tool)} "
            f"{shlex.quote(pod_input)} --expected-report "
            f"{shlex.quote(f'{pod_output_dir}/source_integrity.json')} "
            ">/dev/null || resume_source_guard_rc=$?; "
            '[ "$resume_source_guard_rc" -eq 0 ]; }'
        )
    compare_source_integrity_step = (
        "source_fingerprint_after=$("
        f"{venv_python} {shlex.quote(fingerprint_tool)} "
        f"{shlex.quote(pod_input)}) || source_integrity_rc=$?; "
        f'printf \'{{"before":"%s","after":"%s","unchanged":%s}}\' '
        '"$source_fingerprint_before" "$source_fingerprint_after" '
        '"$(if [ "$source_fingerprint_before" = "$source_fingerprint_after" ]; '
        'then printf true; else printf false; fi)" '
        f"> {shlex.quote(f'{pod_output_dir}/source_integrity.json')}; "
        'if [ "$source_integrity_rc" -eq 0 ] && '
        '[ "$source_fingerprint_before" != "$source_fingerprint_after" ]; '
        "then source_integrity_rc=90; fi"
    )
    source_integrity_step = (
        'if [ "${resume_source_guard_rc:-0}" -ne 0 ]; then '
        'source_integrity_rc="$resume_source_guard_rc"; else '
        f"source_integrity_rc=0; {compare_source_integrity_step}; fi"
    )
    pipeline_step = (
        f"{env_prefix} {venv_python} {shlex.quote(frozen_script)}"
        f" {' '.join(pipeline_args)}"
        f" > {shlex.quote(pipeline_log)} 2>&1"
    )
    validation_step = "validation_rc=0"
    if output_schema is not None and output_schema != "structured":
        validator = f"{pod_runtime_dir}/validate_prepared_data.py"
        validation_report = f"{pod_output_dir}/validation.json"
        image_root_arg = (
            '--image-root "$image_root" ' if output_schema == "vision" else ""
        )
        image_root_step = (
            f"if [ -d {shlex.quote(pod_input)} ]; then "
            f"image_root={shlex.quote(pod_input)}; else "
            f"image_root=$(dirname {shlex.quote(pod_input)}); fi; "
        )
        if output_schema == "vision":
            image_root_source = """\
import json
import sys
from pathlib import Path

source, metadata_path = map(Path, sys.argv[1:])
root = str(source if source.is_dir() else source.parent)
if metadata_path.is_file():
    metadata = json.loads(metadata_path.read_text())
    if (metadata.get("output_format") == "vision"
            and metadata.get("source_path") == str(source.resolve())):
        root = metadata["image_root"]
print(root)
"""
            image_root_step = (
                "image_root=$("
                + shlex.join(
                    [
                        venv_python,
                        "-c",
                        image_root_source,
                        pod_input,
                        f"{pod_output_dir}/runtime_metadata.json",
                    ]
                )
                + "); "
            )
        validation_step = (
            "validation_rc=0; "
            'if [ "$pipeline_rc" -eq 0 ]; then '
            f"{image_root_step}"
            f"{venv_python} {shlex.quote(validator)} "
            f"{shlex.quote(output_file)} --schema {shlex.quote(output_schema)} "
            f"{image_root_arg}"
            f"--report {shlex.quote(validation_report)}"
            f" > {shlex.quote(validation_log)} 2>&1"
            " || validation_rc=$?; fi"
        )
    report_step = (
        f"{venv_python} {shlex.quote(pod_runtime_dir)}/generate_report.py"
        f" --log-dir {shlex.quote(pod_output_dir)}"
        f' --pipeline-exit-code "$pipeline_rc"'
        f" --execution-revision {execution_revision}"
        f" --resumed {'true' if resumed else 'false'}"
        f" --runtime-dir-name {shlex.quote(runtime_dir_name)}"
    )
    if image_utils_api_version:
        report_step += " --image-utils-api-version " + shlex.quote(
            image_utils_api_version
        )
    if runtime_fingerprint:
        report_step += " --runtime-fingerprint " + shlex.quote(runtime_fingerprint)
    if reuse_assessment is not None:
        report_step += " --reuse-assessment-json " + shlex.quote(
            json.dumps(reuse_assessment, ensure_ascii=False)
        )
    report_step += f" >> {shlex.quote(report_log)} 2>&1 || true"
    final_step = (
        'if [ "$source_integrity_rc" -ne 0 ]; then '
        f'echo "Data processing failed during source integrity check; '
        f'see {pod_output_dir}/source_integrity.json" >&2; '
        'exit "$source_integrity_rc"; fi; '
        'if [ "$pipeline_rc" -ne 0 ]; then '
        f'echo "Data processing failed with exit code $pipeline_rc; '
        f'last log lines:" >&2; tail -n 40 {shlex.quote(pipeline_log)} >&2; '
        'exit "$pipeline_rc"; fi; '
        'if [ "$validation_rc" -ne 0 ]; then '
        f'echo "Output validation failed; '
        f'last log lines:" >&2; tail -n 40 {shlex.quote(validation_log)} >&2; '
        'exit "$validation_rc"; fi; '
        f'echo "Data preparation completed: {output_file}"'
    )
    setup = " && ".join(setup_steps)
    setup_guard = (
        f"{{ {setup}; }} || {{ setup_rc=$?; "
        "echo 'Data processing setup failed; last setup log lines:' >&2; "
        f'tail -n 40 {shlex.quote(setup_log)} >&2; exit "$setup_rc"; }}'
    )
    return (
        f"{setup_guard}; {pipeline_step}"
        + f"; pipeline_rc=$?; {validation_step}; {source_integrity_step}; "
        + f"{report_step}; {final_step}"
    )


def _build_dataflow_workflow(
    action: PipelineOptions,
    run_id: uuid.UUID,
    command: str,
) -> dict[str, Any]:
    return {
        "id": str(run_id),
        "name": f"agent-data-prep-{str(run_id)[:8]}",
        "nodes": [
            {
                "id": "1",
                "type": "default",
                "position": {"x": 0, "y": 0},
                "data": {
                    "display_name": "Data Processing Pipeline",
                    "nodeType": DATAFLOW_NODE_TYPE,
                    "config": {
                        "command": command,
                        "cpu": action.cpu,
                        "memory": action.memory,
                    },
                },
            }
        ],
        "edges": [],
        "viewport": {"x": 0, "y": 0, "zoom": 1},
        "timestamp": datetime.now(UTC).isoformat(),
    }


MODEL_PREPARATION_NODES = frozenset(
    {"CloneAndCacheModel", "DownloadAndCacheModel", "ModelMergeLoraNode"}
)


def validate_pipeline_topology(
    workflow: dict[str, Any], *, require_inference: bool = False
) -> None:
    nodes, edges = workflow.get("nodes", []), workflow.get("edges", [])
    kinds = [node["data"]["nodeType"] for node in nodes]
    if not require_inference and kinds == ["CustomCommandCPUNode"] and not edges:
        return
    preparation = [
        node for node in nodes if node["data"]["nodeType"] in MODEL_PREPARATION_NODES
    ]
    expected = ["CustomCommandCPUNode", "VLLMInference"]
    if len(preparation) == 1:
        expected.append(preparation[0]["data"]["nodeType"])
    if sorted(kinds) != sorted(expected) or len(edges) != len(nodes) - 1:
        raise ValueError(
            "evaluation requires inference -> CPU with at most one "
            "clone or download node"
        )
    if len({str(node["id"]) for node in nodes}) != len(nodes):
        raise ValueError("evaluation node IDs must be unique")
    source = next(node for node in nodes if node["data"]["nodeType"] == "VLLMInference")
    target = next(
        node for node in nodes if node["data"]["nodeType"] == "CustomCommandCPUNode"
    )
    expected_edges = {(str(source["id"]), str(target["id"]), "endpoint", "param")}
    if preparation:
        expected_edges.add(
            (
                str(preparation[0]["id"]),
                str(source["id"]),
                "merged_model_path"
                if preparation[0]["data"]["nodeType"] == "ModelMergeLoraNode"
                else "model_path",
                "model_path",
            )
        )
    actual_edges = {
        (
            str(edge.get("source")),
            str(edge.get("target")),
            edge.get("sourceHandle"),
            edge.get("targetHandle"),
        )
        for edge in edges
    }
    if actual_edges != expected_edges:
        raise ValueError(
            "pipeline must connect preparation.model_path to inference.model_path "
            "and inference.endpoint to CPU.param"
        )


MAX_CLEANING_SCRIPT_BYTES = 1024 * 1024


def _runtime_public_names(utils_path: Path) -> set[str]:
    try:
        tree = ast.parse(
            utils_path.read_text(encoding="utf-8"),
            filename=str(utils_path),
        )
    except (OSError, SyntaxError) as exc:
        raise ValueError(f"Could not inspect cleaning_utils public API: {exc}") from exc
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in targets
        ):
            continue
        if node.value is None:
            raise ValueError("cleaning_utils __all__ must have a literal value")
        try:
            value = ast.literal_eval(node.value)
        except (TypeError, ValueError) as exc:
            raise ValueError("cleaning_utils __all__ must be a literal list") from exc
        if not isinstance(value, list) or not all(
            isinstance(name, str) for name in value
        ):
            raise ValueError("cleaning_utils __all__ must contain only strings")
        return set(value)
    raise ValueError("cleaning_utils does not define a public __all__ contract")


def validate_cleaning_script(
    source: str, utils_path: Path, *, filename: str = "clean_script.py"
) -> None:
    """Check syntax and runtime imports without executing the cleaning script."""
    try:
        tree = ast.parse(source, filename=filename)
        compile(tree, filename, "exec")
    except SyntaxError as exc:
        location = f"line {exc.lineno}"
        if exc.offset is not None:
            location += f", column {exc.offset}"
        raise ValueError(
            f"Dataset cleaning script syntax error at {location}: {exc.msg}"
        ) from exc
    public_names = _runtime_public_names(utils_path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(
            alias.name == "cleaning_utils" for alias in node.names
        ):
            raise ValueError(
                "Dataset cleaning script must use explicit "
                "from cleaning_utils import ... APIs."
            )
        if not isinstance(node, ast.ImportFrom) or node.module != "cleaning_utils":
            continue
        imported = {alias.name for alias in node.names}
        if "*" in imported:
            raise ValueError(
                "Dataset cleaning script must import explicit cleaning_utils APIs."
            )
        unknown = sorted(imported - public_names)
        if unknown:
            raise ValueError(
                "Dataset cleaning script imports unsupported cleaning_utils APIs: "
                + ", ".join(unknown)
            )


def _build_cleaning_command(
    *,
    input_path: str,
    script_path: str,
    output_root: str,
    output_dir: str,
    limit: int | None,
    resumed: bool,
    frozen_script_ready: bool = False,
) -> str:
    pod_input = _pod_path(input_path)
    pod_output_root = _pod_path(output_root)
    pod_output_dir = _pod_path(output_dir)
    frozen_script = f"{pod_output_dir}/clean_script.py"
    cleaning_utils = f"{pod_output_dir}/cleaning_utils.py"
    validator = f"{pod_output_dir}/validate_format.py"
    output_file = f"{pod_output_dir}/output.jsonl"
    report_file = f"{pod_output_dir}/report.json"

    command_parts = [
        "python3",
        shlex.quote(frozen_script),
        "--input",
        shlex.quote(pod_input),
        "--output",
        shlex.quote(output_file),
        "--state-dir",
        shlex.quote(pod_output_dir),
    ]
    if resumed:
        command_parts.append("--resume")
    if resumed or frozen_script_ready:
        prefix = " && ".join(
            f"test -f {shlex.quote(path)}"
            for path in (frozen_script, cleaning_utils, validator)
        )
    else:
        prefix = " && ".join(
            [
                f"mkdir -p {shlex.quote(pod_output_root)}",
                f"mkdir -p {shlex.quote(pod_output_dir)}",
                (
                    f"cp {shlex.quote(_pod_path(script_path))} "
                    f"{shlex.quote(frozen_script)}"
                ),
            ]
        )
    if limit is not None:
        command_parts.extend(["--limit", str(limit)])
    validator_parts = [
        "python3",
        shlex.quote(validator),
        "--input",
        shlex.quote(output_file),
        "--report",
        shlex.quote(report_file),
    ]
    return f"{prefix} && {' '.join(command_parts)} && {' '.join(validator_parts)}"


def _build_cleaning_workflow(
    action: CleaningOptions,
    run_id: uuid.UUID,
    command: str,
) -> dict[str, Any]:
    return {
        "id": str(run_id),
        "name": f"agent-data-clean-{str(run_id)[:8]}",
        "nodes": [
            {
                "id": "1",
                "type": "default",
                "position": {"x": 0, "y": 0},
                "data": {
                    "display_name": "Custom Command",
                    "nodeType": "CustomCommandCPUNode",
                    "config": {
                        "command": command,
                        "cpu": action.cpu,
                        "memory": action.memory,
                    },
                },
            }
        ],
        "edges": [],
        "viewport": {"x": 0, "y": 0, "zoom": 1},
        "timestamp": datetime.now(UTC).isoformat(),
    }


def build_edp_workflow(
    *,
    run_id: uuid.UUID,
    command: str,
    cpu: int,
    memory: int,
) -> dict[str, Any]:
    return {
        "id": str(run_id),
        "name": f"agent-edp-{str(run_id)[:8]}",
        "nodes": [
            {
                "id": "1",
                "type": "default",
                "position": {"x": 0, "y": 0},
                "data": {
                    "display_name": "Env Validation Batch",
                    "nodeType": DATAFLOW_NODE_TYPE,
                    "config": {"command": command, "cpu": cpu, "memory": memory},
                },
            }
        ],
        "edges": [],
        "viewport": {"x": 0, "y": 0, "zoom": 1},
        "timestamp": datetime.now(UTC).isoformat(),
    }


RENDER_FILENAME = "render_manifest.py"


RENDER_TEMPLATE_FILENAME = "render_template.json"


RENDER_REQUIREMENTS = "render_requirements.txt"


def _spec_column_refs(name: str, spec: Any) -> list[tuple[str, str]]:
    """Collect (field, column) references a template spec reads from the row."""
    if isinstance(spec, str):
        return [(name, spec)]
    if not isinstance(spec, dict):
        return []
    if "field" in spec:
        return [(name, str(spec["field"]))]
    kind = spec.get("kind")
    if kind == "message":
        return [(name, str(spec.get("source_field", "messages")))]
    if kind == "pytest_wrapper":
        source = spec.get("source", spec.get("source_field"))
        if source is not None:
            return _spec_column_refs(name, source)
    return []


def _collect_column_refs(fields: dict[str, Any]) -> list[tuple[str, str]]:
    refs: list[tuple[str, str]] = []
    for name, spec in fields.items():
        refs.extend(_spec_column_refs(name, spec))
    return refs


def _check_column_refs(schema: Any, refs: list[tuple[str, str]]) -> list[str]:
    import pyarrow as pa

    errors: list[str] = []
    top_names = list(schema.names)
    for field_name, column in refs:
        parts = column.split(".")
        if parts[0] not in top_names:
            errors.append(
                f"field {field_name!r} references top-level column "
                f"{parts[0]!r} which does not exist (available columns: "
                f"{top_names})"
            )
            continue
        arrow_field = schema.field(parts[0])
        for depth, part in enumerate(parts[1:], start=1):
            field_type = arrow_field.type
            if not pa.types.is_struct(field_type):
                errors.append(
                    f"field {field_name!r}: column {column!r} descends into "
                    f"{part!r} but {parts[depth - 1]!r} is {field_type}, "
                    "not a struct"
                )
                break
            if part not in field_type.names:
                errors.append(
                    f"field {field_name!r}: column {column!r} references "
                    f"missing nested field {part!r} (struct fields: "
                    f"{list(field_type.names)})"
                )
                break
            arrow_field = field_type.field(part)
    return errors


def build_render_command(
    *,
    output_dir: str,
    data_source: str,
    shard_size: int,
    limit: int | None,
    cluster: str | None = None,
) -> str:
    """Assemble the render command: install minimal deps, run render_manifest."""
    pod_output = _pod_path(to_tool_storage_path(output_dir, field="output_dir"))
    venv = "/tmp/edp-venv"
    data_pod = _pod_path(to_tool_storage_path(data_source, field="data_source"))
    out_pod = pod_output

    steps = [
        f"python3 -m venv {venv}",
        (
            _pip_install_command(
                pip_path=f"{venv}/bin/pip",
                arguments=[
                    "--quiet",
                    "--use-deprecated=legacy-resolver",
                    "-r",
                    f"{pod_output}/{RENDER_REQUIREMENTS}",
                ],
                cluster=cluster,
            )
            + " > /tmp/edp-render-install.log 2>&1"
            " || { echo 'render dependencies install failed; "
            "last install log lines:'; tail -n 100 "
            "/tmp/edp-render-install.log; exit 1; }"
        ),
        f"test -f {pod_output}/{RENDER_FILENAME}",
        f"test -f {pod_output}/{RENDER_TEMPLATE_FILENAME}",
        (
            f"{venv}/bin/python {pod_output}/{RENDER_FILENAME} "
            f"--template {pod_output}/{RENDER_TEMPLATE_FILENAME} "
            f"--data-source {shlex.quote(data_pod)} "
            f"--output-root {shlex.quote(out_pod)} "
            f"--shard-size {int(shard_size)}"
        )
        + (f" --limit {int(limit)}" if limit is not None else ""),
    ]
    return " && ".join(steps)


AGGREGATE_FILENAME = "aggregate_results.py"


SLIME_CONVERTER_FILENAME = "convert_to_slime.py"


SFT_CONVERTER_FILENAME = "convert_to_sft.py"


def build_aggregate_command(
    *,
    out_dir: str,
    run_dirs: Sequence[str],
    protocol: str,
    min_reward: float,
    system_prompt: str | None,
    limit: int | None,
) -> str:
    """Assemble the aggregation command: node python3, stdlib only."""
    pod_out = _pod_path(str(PurePosixPath("/" + out_dir.lstrip("/"))))
    pod_run_dirs = [
        _pod_path(str(PurePosixPath("/" + str(d).lstrip("/")))) for d in run_dirs
    ]
    script = f"{pod_out}/{AGGREGATE_FILENAME}"
    args = [
        "--run-dirs",
        *[shlex.quote(d) for d in pod_run_dirs],
        "--out-dir",
        shlex.quote(pod_out),
        "--protocol",
        shlex.quote(protocol),
        "--min-reward",
        str(float(min_reward)),
    ]
    if system_prompt is not None:
        args += ["--system-prompt", shlex.quote(system_prompt)]
    if limit is not None:
        args += ["--limit", str(int(limit))]
    steps = [
        # Leading no-op: the pod executor can drop the first command segment.
        "true",
        f"test -f {script}",
        f"test -f {pod_out}/{SLIME_CONVERTER_FILENAME}",
        f"test -f {pod_out}/{SFT_CONVERTER_FILENAME}",
        # Pure standard library: run with the node python directly. The
        # script's own directory is sys.path[0], so its sibling converter
        # imports resolve from the same staged directory.
        f"python3 {script} " + " ".join(args),
    ]
    return " && ".join(steps)


def prepare_debug_workflow(workflow: dict[str, Any]) -> dict[str, Any]:
    """Copy a prepared graph and force test execution without mutating its source."""
    prepared = deepcopy(workflow)
    execution_argos = prepared.get("execution_argos") or []
    if not isinstance(execution_argos, list):
        raise ValueError("execution_argos must be a list")
    execution_argos = [
        item
        for item in execution_argos
        if not isinstance(item, dict) or "execution_mode" not in item
    ]
    prepared["execution_argos"] = [*execution_argos, {"execution_mode": "test"}]
    return prepared
