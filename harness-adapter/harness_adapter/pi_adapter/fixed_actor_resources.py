"""Prepare the Agent's current resources using an existing Actor image and mount."""

from __future__ import annotations

import shlex
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path, PurePosixPath

from harness_adapter.pi_adapter.actor_workspace import ActorWorkspace
from harness_adapter.pi_adapter.platform_helper_bundle import (
    build_platform_helper_wheel,
)
from harness_adapter.pi_adapter.sandbox_runtime import (
    PYROMIND_AGENT_RESOURCES_ROOT,
    _resource_archive,
    _resource_digest,
)
from harness_adapter.pi_adapter.sdk_environment import PLATFORM_SDK_VERSION


_IMAGE_RESOURCES = "/opt/pyromind/resources"


def _prepare_cached_file(
    workspace: ActorWorkspace,
    remote: PurePosixPath,
    content: Callable[[], bytes],
    command: list[str],
    seed: list[str] | None = None,
) -> None:
    parent = shlex.quote(str(remote.parent))
    path = shlex.quote(str(remote))
    on_miss = shlex.join(seed) if seed else ":"
    output = workspace.checked_command(
        f"mkdir -p -- {parent} && if test -f {path}; then echo cached; "
        f"else {on_miss}; fi",
        timeout=120,
    )
    activate = shlex.join(command)
    if output.strip() == "cached":
        try:
            workspace.checked_command(activate, timeout=120)
            return
        except RuntimeError:
            pass
    workspace.transport.write_file(str(remote), content())
    workspace.checked_command(activate, timeout=120)


_ACTOR_RESOURCES = """\
import fcntl
import hashlib
import re
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

cache, version = sys.argv[1:3]
archive = Path(cache) / (version + ".tgz")
marker = Path(".pyromind-actor-resource-version")

def hash_file(checksum, name, source):
    checksum.update(name.encode() + b"\\0")
    with source:
        for block in iter(lambda: source.read(1 << 20), b""):
            checksum.update(block)
    checksum.update(b"\\0")

if len(sys.argv) == 4:
    image = Path(sys.argv[3])
    paths = sorted(
        path.relative_to(image).as_posix()
        for root in (image / ".agents/skills", image / "knowledge")
        for path in root.rglob("*")
        if not path.is_symlink() and path.is_file() and "__pycache__" not in path.parts
    )
    digest = hashlib.sha256()
    for name in paths:
        hash_file(digest, name, (image / name).open("rb"))
    if digest.hexdigest()[:32] == version:
        with tempfile.TemporaryDirectory(dir=cache) as staging:
            staged = Path(staging) / "resources.tgz"
            with tarfile.open(staged, "w:gz", dereference=True) as bundle:
                for name in paths:
                    bundle.add(image / name, arcname=name)
            staged.replace(archive)
        print("cached")
    sys.exit(0)

def resource_names(path, digest):
    result = set()
    checksum = hashlib.sha256()
    with tarfile.open(path) as bundle:
        for entry in bundle:
            dest = Path(entry.name)
            if not dest.resolve().is_relative_to(Path.cwd()):
                raise ValueError("Resource must be inside the conversation workspace")
            if not (entry.isfile() or entry.islnk()):
                raise ValueError("Resource must be a regular file")
            result.add(entry.name)
            hash_file(checksum, entry.name, bundle.extractfile(entry))
    if checksum.hexdigest()[:32] != digest:
        raise ValueError("Resource archive checksum mismatch")
    return result

with open(str(marker) + ".lock", "a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    previous = marker.read_text(errors="replace").strip() if marker.is_file() else ""
    if previous != version:
        current = resource_names(archive, version)
        versions = {v for v in previous.split() if re.fullmatch("[0-9a-f]{32}", v)}
        marker.write_text("updating " + " ".join(versions | {version}))
        for digest in versions - {version}:
            try:
                old = resource_names(archive.with_name(digest + ".tgz"), digest)
            except (OSError, EOFError, tarfile.TarError, ValueError):
                continue
            for name in old - current:
                Path(name).unlink(missing_ok=True)
        subprocess.run(["tar", "--unlink-first", "--no-same-owner",
                        "--no-same-permissions", "-xzf", archive], check=True)
        marker.write_text(version)
"""


def prepare_actor_resources(
    workspace: ActorWorkspace, resource_files: Sequence[tuple[str, str]]
) -> None:
    if not resource_files:
        return
    files = sorted(dict(resource_files).items())
    version = _resource_digest(files, workspace.working_dir)
    cache = PurePosixPath(workspace.storage_path + PYROMIND_AGENT_RESOURCES_ROOT)
    archive = cache / (version + ".tgz")

    def content() -> bytes:
        with _resource_archive(files, workspace.working_dir) as local:
            return local.read_bytes()

    command = ["python3", "-I", "-c", _ACTOR_RESOURCES, str(cache), version]
    _prepare_cached_file(
        workspace, archive, content, command, [*command, _IMAGE_RESOURCES]
    )


_PREPARE_SDK = """\
import fcntl
import site
import subprocess
import sys
import venv
import zipfile
from pathlib import Path

directory, wheel, sdk_version, helper_version = sys.argv[1:]
root = Path(directory)
python = root / "bin/python"
with open(directory + ".lock", "a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    ready = root / ".ready"
    if not (ready.is_file() and python.is_file()):
        venv.EnvBuilder(symlinks=True, with_pip=False).create(root)
        packages = Path(site.getsitepackages([directory])[0])
        with zipfile.ZipFile(wheel) as bundle:
            bundle.extractall(packages)
        paths = [site.getsitepackages(["/opt/pyromind/" + env])[0]
                 for env in ("sdk-venv", "dataflow-venv")]
        (packages / "actor-image.pth").write_text("\\n".join(paths) + "\\n")
        check = (
            "import sys; from importlib.metadata import version; "
            "from pyromind_sdk import PyroMindAPIClient; "
            "from pyromind_common.platform import workflows, preparation, "
            "evaluation, inspection, progress; "
            "assert version('pyromind-sdk') == sys.argv[1]; "
            "assert version('pyromind-platform-helpers') == sys.argv[2]"
        )
        subprocess.run([python, "-I", "-c", check, sdk_version, helper_version],
                       check=True)
        ready.touch()
"""


def prepare_actor_sdk(workspace: ActorWorkspace) -> str:
    with tempfile.TemporaryDirectory() as directory:
        bundle = build_platform_helper_wheel(Path(directory))
        remote = (
            PurePosixPath(workspace.storage_path + PYROMIND_AGENT_RESOURCES_ROOT)
            / bundle.path.name
        )
        sdk_dir = f"/tmp/pyromind-actor-sdk-{PLATFORM_SDK_VERSION}-{bundle.version}"
        _prepare_cached_file(
            workspace,
            remote,
            bundle.path.read_bytes,
            [
                "/opt/pyromind/sdk-venv/bin/python",
                "-I",
                "-c",
                _PREPARE_SDK,
                sdk_dir,
                str(remote),
                PLATFORM_SDK_VERSION,
                bundle.version,
            ],
        )
    return sdk_dir + "/bin/python"
