# PyroMind Knowledge Base

Agent 生产运行时通过 `PYROMIND_KNOWLEDGE_BASE_PATH`（默认指向本目录）检索文档。

## 目录结构

| 路径 | 内容 |
|------|------|
| `basic/`、`sdk/`、`studio/`、`jupyterlab/` | 内嵌平台文档（Agent 运行时检索根目录） |
| [`../docs-mintlify/`](../docs-mintlify/) | **git submodule** — 官方 Mintlify 文档源（[上游仓库](https://github.com/PyroMind-Dynamics/docs-mintlify)），位于仓库根目录 |

[PyroMind Python SDK](https://pypi.org/project/pyromind-sdk/) 通过 PyPI 安装，`make build`（`uv sync --dev`）会默认安装。

## Agent 检索路径

**平台使用文档（内嵌）：**

- `basic/` — 基础用法
- `sdk/` — SDK 文档
- `studio/` — Studio 文档
- `jupyterlab/` — JupyterLab 文档

**Mintlify 官方文档（submodule，仓库根目录）：**

- `docs-mintlify/zh/docs/` — 中文文档
- `docs-mintlify/en/docs/` — 英文文档

## 首次克隆后初始化 submodule

```bash
git submodule update --init --recursive docs-mintlify
```

## 更新上游文档（提升 submodule 指针）

```bash
cd docs-mintlify
git fetch origin && git checkout <commit-or-branch>
cd ..

git add docs-mintlify
git commit -m "Bump docs-mintlify submodule"
```

## Docker 构建

根目录 `Dockerfile` 会 `COPY knowledge ./knowledge` 与 `COPY docs-mintlify ./docs-mintlify`。构建镜像前必须先初始化 submodule：

```bash
git submodule update --init --recursive docs-mintlify
docker build -t pyromind-agent-server .
```
