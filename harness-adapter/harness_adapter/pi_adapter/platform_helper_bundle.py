"""Package the terminal's pure platform helpers without the Host dependency tree."""

from __future__ import annotations

import ast
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from zipfile import ZipInfo

from wheel.wheelfile import WheelFile

from harness_adapter.pi_adapter.sdk_environment import (
    PLATFORM_SDK_VERSION,
    platform_sdk_bootstrap_source,
)


HELPER_DISTRIBUTION = "pyromind-platform-helpers"
HELPER_SOURCE_FILES = (
    "__init__.py",
    "platform/__init__.py",
    "platform/paths.py",
    "platform/credentials.py",
    "platform/inspection.py",
    "platform/progress.py",
    "platform/workflows.py",
    "platform/workflow_paths.py",
    "platform/scene_options.py",
    "platform/scene_core.py",
    "platform/recipe.py",
    "platform/preparation.py",
    "platform/environment.py",
    "platform/evaluation_core.py",
    "platform/evaluation.py",
)
_REQUIREMENTS = (
    f"pyromind-sdk=={PLATFORM_SDK_VERSION}",
    "httpx>=0.27.0",
    "pydantic>=2.0.0",
    "requests>=2.28.0",
    "filelock>=3.20.1",
)


@dataclass(frozen=True)
class PlatformHelperWheel:
    path: Path
    version: str


def build_platform_helper_wheel(directory: Path) -> PlatformHelperWheel:
    root = files("pyromind_common")
    sources: dict[str, bytes] = {}
    digest = hashlib.sha256("\n".join(_REQUIREMENTS).encode())
    for relative in HELPER_SOURCE_FILES:
        source = root.joinpath(relative).read_bytes()
        ast.parse(source, filename=relative, feature_version=(3, 10))
        name = "pyromind_common/" + relative
        sources[name] = source
        digest.update(name.encode() + b"\0" + source + b"\0")
    version = "0.1.0+" + digest.hexdigest()[:16]
    distribution = HELPER_DISTRIBUTION.replace("-", "_")
    metadata_root = f"{distribution}-{version}.dist-info"
    metadata = (
        "Metadata-Version: 2.4\n"
        f"Name: {HELPER_DISTRIBUTION}\n"
        f"Version: {version}\n"
        "Summary: Pure platform preparation helpers for Agent terminals\n"
        "Requires-Python: >=3.10\n"
        + "".join(f"Requires-Dist: {requirement}\n" for requirement in _REQUIREMENTS)
        + "\n"
    )
    sources[f"{metadata_root}/METADATA"] = metadata.encode()
    sources[f"{metadata_root}/WHEEL"] = (
        b"Wheel-Version: 1.0\n"
        b"Generator: pyromind-platform-helper-bundle\n"
        b"Root-Is-Purelib: true\n"
        b"Tag: py3-none-any\n"
    )
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{distribution}-{version}-py3-none-any.whl"
    with WheelFile(path, "w") as wheel:
        for name, data in sorted(sources.items()):
            info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o644 << 16
            wheel.writestr(info, data)
    return PlatformHelperWheel(path=path, version=version)


def prepare_host_sdk() -> str:
    """Install the same pure helper wheel in an interpreter outside Host source."""
    executable = shutil.which("python3") if vars(sys).get("frozen") else sys.executable
    if not executable:
        raise RuntimeError("A real Python interpreter is required for platform SDK use")
    with tempfile.TemporaryDirectory(prefix="pyromind-platform-helpers-") as staging:
        bundle = build_platform_helper_wheel(Path(staging))
        directory = Path(tempfile.gettempdir()) / (
            f"pyromind-platform-sdk-{PLATFORM_SDK_VERSION}-"
            + bundle.version.partition("+")[2]
        )
        source = platform_sdk_bootstrap_source(
            helper_wheel=str(bundle.path),
            helper_version=bundle.version,
            sdk_directory=str(directory),
        )
        environment = {
            name: value
            for name, value in os.environ.items()
            if name
            in {
                "PATH",
                "HOME",
                "TMPDIR",
                "LANG",
                "LC_ALL",
                "HTTP_PROXY",
                "HTTPS_PROXY",
                "NO_PROXY",
                "http_proxy",
                "https_proxy",
                "no_proxy",
                "SSL_CERT_FILE",
                "SSL_CERT_DIR",
                "PIP_INDEX_URL",
                "PIP_EXTRA_INDEX_URL",
                "PIP_TRUSTED_HOST",
            }
        }
        result = subprocess.run(
            [executable, "-c", source],
            env=environment,
            capture_output=True,
            text=True,
            timeout=360,
            check=False,
        )
        if result.returncode:
            raise RuntimeError(
                "Platform SDK interpreter preparation failed "
                f"(exit_code={result.returncode})"
            )
    return str(directory / "bin" / "python")
