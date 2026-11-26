"""Prepare the pinned platform SDK in the sandbox execution plane."""

PLATFORM_SDK_VERSION = "0.1.17"
SANDBOX_SDK_DIRECTORY = f"/tmp/pyromind-platform-sdk-{PLATFORM_SDK_VERSION}"
SANDBOX_SDK_PYTHON = f"{SANDBOX_SDK_DIRECTORY}/bin/python"
SDK_ATOMIC_TOOL_NAMES = frozenset(
    {
        "sandbox_create",
        "sandbox_delete",
        "sandbox_read_file",
        "sandbox_write_file",
        "sandbox_delete_file",
        "sandbox_terminal",
        "df_stop_task",
        "inspect_workflow_task",
    }
)
SDK_REMOTE_STORAGE_TOOL_NAMES = frozenset(
    {
        "upload_file_to_pyromind",
        "sandbox_upload",
        "sandbox_download",
        "df_check_progress",
    }
)
PLATFORM_SDK_INSTRUCTIONS = """\
Platform SDK access is enabled for this main terminal. Run SDK Python scripts
with "$PYROMIND_SDK_PYTHON"; authentication, base URL and cluster are supplied
automatically. SDK requests use the current signed-in platform account permissions.
For data preparation and workflow submission, use these Python imports:
from pyromind_common.platform.preparation import prepare_pipeline
from pyromind_common.platform.workflows import submit_workflow
For Studio submissions, use PyroMindAPIClient(max_retries=0) and an unchanged out_id
from the JSON PYROMIND_TASK_CONTEXTS index for the matching task kind. For approved
Project work, select the matching task_id and attempt in PYROMIND_PROJECT_EXECUTIONS
and use its out_ids entry. Do not construct or modify signed out_id values.
Read knowledge/sdk/agent-operations.mdx and the matching skill's preparation and
output contracts as needed. Do not automatically repeat create after an uncertain
result; retain any returned task_id and reconcile it before further submission.
Use PyroMindAPIClient().sandboxes for sandbox lifecycle, file
operations and command execution, and .studio.stop(task_id) to request a task stop.
For sandbox creation without a user-specified image, pass the image supplied by
PYROMIND_SANDBOX_DEFAULT_IMAGE. Follow the sandbox skill's SDK references for
request models, bounded output, and execution status. When sandbox_terminal is
listed, use it for sandbox commands. A stop response acknowledges the request;
query the task to confirm termination.
For workflow task and node status or bounded node logs, use the ordinary Python
inspection helpers described in knowledge/sdk/agent-operations.mdx. They query
the SDK directly and do not submit tasks or create Project execution records.
In the remote sandbox, Storage is already mounted: publish and read files through
the actual mount rather than upload/download tools. Read progress.json and report
files in the known run output directory; use get_storage_url for artifact links.
"""


class PlatformSDKSetupError(RuntimeError):
    def __init__(self, stage: str, exit_code: int | None, timed_out: bool) -> None:
        self.stage = stage
        self.exit_code = exit_code
        self.timed_out = timed_out
        super().__init__(
            f"PI_PLATFORM_SDK_SETUP_FAILED stage={stage} "
            f"exit_code={exit_code} timed_out={timed_out}. "
            f"Setup log: {SANDBOX_SDK_DIRECTORY}.bootstrap.log"
        )


def platform_sdk_bootstrap_source(
    *,
    helper_wheel: str | None = None,
    helper_version: str | None = None,
    sdk_directory: str = SANDBOX_SDK_DIRECTORY,
    system_python: str | None = None,
    pip_index_url: str | None = None,
) -> str:
    if (helper_wheel is None) != (helper_version is None):
        raise ValueError("The helper wheel and its version must be supplied together")
    requirements = [f"pyromind-sdk=={PLATFORM_SDK_VERSION}"]
    check = (
        "import pyromind_sdk; "
        "from importlib.metadata import version; "
        f"assert version('pyromind-sdk') == {PLATFORM_SDK_VERSION!r}"
    )
    if helper_wheel is not None:
        requirements.append(helper_wheel)
        check += (
            "; from pyromind_common.platform import workflows, preparation, "
            "evaluation, inspection, progress; "
            f"assert version('pyromind-platform-helpers') == {helper_version!r}"
        )
    return f"""\
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading

directory = Path({sdk_directory!r})
python = directory / "bin" / "python"
check = {check!r}
requirements = {requirements!r}
system_python = {system_python!r}
pip_index_url = {pip_index_url!r}
stage = "lock"
stop_heartbeat = threading.Event()

def heartbeat():
    while not stop_heartbeat.wait(10):
        print("PYROMIND_SDK_SETUP: " + stage, flush=True)

def compatible(executable):
    try:
        result = subprocess.run(
            [executable, "-c",
             "import sys, venv, ensurepip; "
             "raise SystemExit(0 if sys.version_info >= (3, 10) else 1)"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10,
        )
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False

def ready():
    if not python.is_file() or not compatible(str(python)):
        return False
    result = subprocess.run(
        [str(python), "-c", check],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30,
    )
    return result.returncode == 0

directory.parent.mkdir(parents=True, exist_ok=True)
thread = threading.Thread(target=heartbeat, daemon=True)
thread.start()
try:
    with Path(str(directory) + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not ready():
            stage = "python"
            candidates = [system_python] if system_python else [
                sys.executable,
                *(shutil.which(name) for name in
                  ("python3.11", "python3.12", "python3.10", "python3")),
                "/usr/bin/python3",
            ]
            executable = next(
                (candidate for candidate in candidates
                 if candidate and compatible(candidate)),
                None,
            )
            if executable is None:
                raise RuntimeError("Python >=3.10 with venv and ensurepip is required")
            stage = "venv"
            log_path = Path(str(directory) + ".bootstrap.log")
            with log_path.open("w") as log:
                os.chmod(log_path, 0o600)
                if not python.is_file() or not compatible(str(python)):
                    subprocess.run(
                        [executable, "-m", "venv", "--clear",
                         "--system-site-packages", str(directory)],
                        stdout=log, stderr=subprocess.STDOUT, check=True, timeout=60,
                    )
                stage = "install"
                print("PYROMIND_SDK_SETUP: install", flush=True)
                environment = dict(os.environ)
                if pip_index_url:
                    environment["PIP_INDEX_URL"] = pip_index_url
                install = [str(python), "-m", "pip", "install",
                           "--disable-pip-version-check", "--progress-bar", "off",
                           "--timeout", "20", "--retries", "2"]
                if pip_index_url:
                    install.extend(["--index-url", pip_index_url])
                subprocess.run(
                    [*install, *requirements],
                    env=environment, stdout=log, stderr=subprocess.STDOUT,
                    check=True, timeout=240,
                )
                stage = "verify"
                if not ready():
                    raise RuntimeError("Platform SDK installation verification failed")
        print("PYROMIND_SDK_SETUP: ready", flush=True)
except Exception as error:
    outcome = ":timeout" if isinstance(error, subprocess.TimeoutExpired) else ""
    print("PYROMIND_SDK_SETUP_ERROR:" + stage + outcome, flush=True)
    raise
finally:
    stop_heartbeat.set()
    thread.join()
"""


def sandbox_sdk_bootstrap_command(
    *,
    helper_wheel: str | None = None,
    helper_version: str | None = None,
    sdk_directory: str = SANDBOX_SDK_DIRECTORY,
    system_python: str | None = None,
    pip_index_url: str | None = None,
) -> str:
    source = platform_sdk_bootstrap_source(
        helper_wheel=helper_wheel,
        helper_version=helper_version,
        sdk_directory=sdk_directory,
        system_python=system_python,
        pip_index_url=pip_index_url,
    )
    return (
        "python3 - <<'PYROMIND_SDK_BOOTSTRAP'\n" + source + "PYROMIND_SDK_BOOTSTRAP\n"
    )
