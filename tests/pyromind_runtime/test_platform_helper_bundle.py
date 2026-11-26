from __future__ import annotations

import csv
import json
import subprocess
import sys
from email.parser import Parser
from io import StringIO
from pathlib import Path
from zipfile import ZipFile

from packaging.requirements import Requirement

from harness_adapter.pi_adapter import platform_helper_bundle as bundle


def test_helper_wheel_installs_and_imports_without_host_packages(tmp_path):
    wheel = bundle.build_platform_helper_wheel(tmp_path / "wheels")
    installed = tmp_path / "installed"
    installed.mkdir()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-deps",
            "--disable-pip-version-check",
            "--target",
            str(installed),
            str(wheel.path),
        ],
        cwd=installed,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    with ZipFile(wheel.path) as archive:
        modules = sorted(
            ".".join(Path(name).with_suffix("").parts)
            for name in archive.namelist()
            if name.startswith("pyromind_common/platform/")
            and name.endswith(".py")
            and not name.endswith("/__init__.py")
        )
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib
import json
import sys
from pathlib import Path

import pyromind_common
from pyromind_common.platform import workflows

root = Path.cwd().resolve()
assert Path(pyromind_common.__file__).resolve().is_relative_to(root)
for name in json.loads(sys.argv[1]):
    module = importlib.import_module(name)
    assert Path(module.__file__).resolve().is_relative_to(root), name
source = {"name": "installed", "execution_argos": []}
prepared = workflows.prepare_debug_workflow(source)
assert prepared["execution_argos"] == [{"execution_mode": "test"}]
assert source["execution_argos"] == []
for name in sys.modules:
    for forbidden in (
        "pyromind_common.tools",
        "pyromind_common.sdk",
        "pyromind_runtime",
        "pyromind_agent_server",
        "harness_adapter",
        "litellm",
        "pyarrow",
    ):
        assert name != forbidden and not name.startswith(forbidden + "."), name
""",
            json.dumps(modules),
        ],
        cwd=installed,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_helper_wheel_metadata_and_record_are_complete(tmp_path):
    wheel = bundle.build_platform_helper_wheel(tmp_path)
    with ZipFile(wheel.path) as archive:
        metadata_path = next(
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        )
        metadata = Parser().parsestr(archive.read(metadata_path).decode())
        assert metadata["Name"] == "pyromind-platform-helpers"
        assert metadata["Version"] == wheel.version
        dependencies = {
            Requirement(value).name for value in metadata.get_all("Requires-Dist", [])
        }
        assert {"pyromind-sdk", "httpx", "pydantic", "requests"} <= dependencies
        assert not dependencies & {
            "pyromind-common",
            "pyromind-runtime",
            "pyromind-agent-server",
            "harness-adapter",
            "litellm",
            "pyarrow",
        }
        record_path = metadata_path.removesuffix("METADATA") + "RECORD"
        records = list(csv.reader(StringIO(archive.read(record_path).decode())))
        assert {row[0] for row in records} == set(archive.namelist())


def test_helper_wheel_version_changes_with_source_bytes(tmp_path, monkeypatch):
    original = bundle.files("pyromind_common")
    copied = tmp_path / "source"
    for relative in bundle.HELPER_SOURCE_FILES:
        destination = copied / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(original.joinpath(relative).read_bytes())
    monkeypatch.setattr(bundle, "files", lambda package: copied)
    first = bundle.build_platform_helper_wheel(tmp_path / "first")
    repeated = bundle.build_platform_helper_wheel(tmp_path / "repeated")
    assert repeated.version == first.version
    workflow = copied / "platform/workflows.py"
    changed_source = workflow.read_bytes() + b"\nBUNDLE_VARIANT = 'updated'\n"
    workflow.write_bytes(changed_source)

    changed = bundle.build_platform_helper_wheel(tmp_path / "changed")

    assert changed.version != first.version
    with ZipFile(changed.path) as archive:
        assert archive.read("pyromind_common/platform/workflows.py") == changed_source
