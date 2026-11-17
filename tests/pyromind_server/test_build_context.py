import shutil
from pathlib import Path

from pyromind_agent_server.docker.build import BuildOptions, _make_build_context


ROOT = Path(__file__).resolve().parents[2]


def test_coexisting_build_context_contains_both_namespaces_and_sandbox_bundles():
    context = _make_build_context(ROOT)
    try:
        for name in (
            "openhands-sdk/pyproject.toml",
            "openhands-tools/pyproject.toml",
            "openhands-agent-server/pyproject.toml",
            "pyromind-common/pyproject.toml",
            "pyromind-embodied-runtime/pyproject.toml",
            "pyromind-agent-server/pyproject.toml",
            ".agents/skills/data-processing/scripts/edp/sandbox_runner.py",
            ".agents/skills/data-processing/scripts/pyromind-edp/sandbox_runner.py",
        ):
            assert (context / name).read_bytes() == (ROOT / name).read_bytes()
        wheels = context / ".agents/skills/data-processing/scripts/pyromind-edp/wheels"
        assert list(wheels.glob("pyromind_embodied_runtime-*.whl"))
        dockerfile = (context / "Dockerfile").read_text()
        assert "uv run pyinstaller pyromind-agent-server/" in dockerfile
        assert "cp /agent-server/dist/pyromind-agent-server" in dockerfile
        stages = [line for line in dockerfile.splitlines() if line.startswith("FROM ")]
        assert stages[-1] == "FROM base-image-minimal AS product"
        assert "FROM base-image-minimal AS binary-minimal" in stages
        assert "FROM base-image-minimal AS pyromind-binary" in stages
    finally:
        shutil.rmtree(context)


def test_product_build_targets_are_explicit_and_do_not_replace_legacy_targets():
    assert BuildOptions().target == "product"
    for name in ("product", "pyromind-binary", "binary-minimal"):
        assert BuildOptions(target=name).target == name
