from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules, copy_metadata
project_root = Path.cwd()
packages = ["pyromind_common", "pyromind_agent_server", "pyromind_runtime", "harness_adapter", "agentgenome", "pyromind_sdk", "litellm", "tiktoken"]
def get_pi_skill_datas():
    data_files = []
    skills_dir = project_root / ".agents" / "skills"
    for skill_name in (
        "inference-evaluation",
        "model-evaluation-preparation",
        "avi-baseline-evaluation",
        "generate-workflow-dsl",
        "data-processing",
        "debug-workflow",
        "sandbox",
        "training-analysis",
        "label-studio",
    ):
        for path in sorted((skills_dir / skill_name).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                rel = path.parent.relative_to(skills_dir).as_posix()
                data_files.append((str(path), f".agents/skills/{rel}"))
    return data_files

datas = get_pi_skill_datas()

binaries = []
hiddenimports = []
for package in packages:
    package_datas, package_binaries, package_imports = collect_all(package)
    datas.extend(package_datas)
    binaries.extend(package_binaries)
    hiddenimports.extend(package_imports)
for dist in ["pyromind-agent-server", "pyromind-common", "pyromind-runtime", "harness-adapter", "pyromind-sdk"]:
    datas.extend(copy_metadata(dist))
a = Analysis([str(project_root / "pyromind-agent-server/pyromind_agent_server/__main__.py")], pathex=[str(project_root / root) for root in ["pyromind-common", "pyromind-runtime", "harness-adapter", "pyromind-agent-server"]], binaries=binaries, datas=datas, hiddenimports=hiddenimports + ["tiktoken_ext.openai_public","uvicorn.logging", "uvicorn.loops.auto", "uvicorn.protocols.http.auto", "uvicorn.protocols.websockets.wsproto_impl", "uvicorn.lifespan.on"], hookspath=[], hooksconfig={}, runtime_hooks=[], excludes=[], noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, a.binaries, a.datas, [], name="pyromind-agent-server", debug=False, bootloader_ignore_signals=False, strip=False, upx=True, console=True)
