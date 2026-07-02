# PyroMind Knowledge Base

Agent 生产运行时通过 `PYROMIND_KNOWLEDGE_BASE_PATH`（默认指向本目录）检索文档。

## 目录结构

| 路径 | 内容 |
|------|------|
| `docs-mintlify/` | **git submodule** — 官方 Mintlify 文档（[上游仓库](https://github.com/PyroMind-Dynamics/docs-mintlify)） |
| `pyromind-sdk-example/` | **git submodule** — 节点 I/O 规范、workflow JSON 样例（[上游仓库](https://github.com/PyroMind-Dynamics/pyromind-sdk-example)） |
| `basic/`、`sdk/`、`studio/`、`jupyterlab/` | 历史内嵌文档（与 `docs-mintlify` 内容可能重叠，优先检索 submodule） |

[PyroMind Python SDK](https://pypi.org/project/pyromind-sdk/) 通过 PyPI 安装，`make build`（`uv sync --dev`）会默认安装。

## Agent 检索路径

**平台使用文档（Mintlify）：**

- `docs-mintlify/zh/docs/` — 中文文档（basic、sdk、studio、jupyterlab）
- `docs-mintlify/en/docs/` — 英文文档

**节点与工作流参考：**

- `pyromind-sdk-example/docs/`、`pyromind-sdk-example/docs_zh/` — 节点定义与端口说明
- `pyromind-sdk-example/workflow/` — 工作流 JSON 样例

## 首次克隆后初始化 submodule

```bash
git submodule update --init --recursive \
  knowledge/docs-mintlify \
  knowledge/pyromind-sdk-example
```

可选：仅检出文档相关目录（减小 `pyromind-sdk-example` 体积）：

```bash
cd knowledge/pyromind-sdk-example
git sparse-checkout init --cone
git sparse-checkout set docs docs_zh workflow imgs
cd ../..
```

## 更新上游文档（提升 submodule 指针）

```bash
# Mintlify 平台文档
cd knowledge/docs-mintlify
git fetch origin && git checkout <commit-or-branch>
cd ../..

# 节点 / workflow 参考
cd knowledge/pyromind-sdk-example
git fetch origin && git checkout <commit-or-branch>
cd ../..

git add knowledge/docs-mintlify knowledge/pyromind-sdk-example
git commit -m "Bump knowledge submodules"
```

## Docker 构建

根目录 `Dockerfile` 会 `COPY knowledge ./knowledge`。构建镜像前必须先初始化 submodule：

```bash
git submodule update --init --recursive \
  knowledge/docs-mintlify \
  knowledge/pyromind-sdk-example
docker build -t pyromind-agent-server .
```
