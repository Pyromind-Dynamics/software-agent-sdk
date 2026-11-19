import shutil
from pathlib import Path

from pyromind_agent_server.docker.build import BuildOptions, _make_build_context


ROOT = Path(__file__).resolve().parents[2]


def test_build_context_contains_only_pyromind_packages_and_canonical_bundle():
    context = _make_build_context(ROOT)
    try:
        for name in (
            "pyromind-common/pyproject.toml",
            "pyromind-embodied-runtime/pyproject.toml",
            "pyromind-agent-server/pyproject.toml",
            ".agents/skills/data-processing/scripts/edp/sandbox_runner.py",
        ):
            assert (context / name).read_bytes() == (ROOT / name).read_bytes()
        wheels = context / ".agents/skills/data-processing/scripts/edp/wheels"
        assert list(wheels.glob("pyromind_embodied_runtime-*.whl"))
        dockerfile = (context / "Dockerfile").read_text()
        assert "uv run pyinstaller pyromind-agent-server/" in dockerfile
        assert "RUN test -x /agent-server/dist/pyromind-agent-server" in dockerfile
        assert not list(context.glob("openhands-*"))
        stages = [line for line in dockerfile.splitlines() if line.startswith("FROM ")]
        assert stages[-1] == "FROM product AS default"
        assert "FROM base-image-minimal AS binary-minimal" in stages
        assert "FROM base-image-minimal AS pyromind-binary" in stages
    finally:
        shutil.rmtree(context)


def test_product_build_targets_are_explicit_and_do_not_replace_legacy_targets():
    assert BuildOptions().target == "product"
    for name in ("product", "pyromind-binary", "binary-minimal"):
        assert BuildOptions(target=name).target == name
