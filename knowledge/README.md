# PyroMind Knowledge Base

Agent 生产运行时通过 `PYROMIND_KNOWLEDGE_BASE_PATH`（默认指向本目录）检索文档。

## 目录结构

| 路径 | 内容 |
|------|------|
| `index.md` | 全量文档目录；检索子 Agent 先按标题、摘要和标签定位候选页面 |
| `basic/`、`sdk/`、`studio/`、`jupyterlab/` | 内嵌平台文档（Agent 运行时检索根目录） |
| `nodes/<NodeType>/<NodeType>.md` | 节点 I/O、参数与端口定义 |
| `business-domain/` | 跨 skill 共用的业务领域知识，如 PCB 裸板 AVI/AOI 数据与判别参考 |
| `dataset_processing_workflow.py` | 工作流 DSL 样例 |

[PyroMind Python SDK](https://pypi.org/project/pyromind-sdk/) 通过 PyPI 安装，`make build`（`uv sync --dev`）会默认安装。

## Agent 检索路径

Agent 首先读取 `index.md`，只在索引缺失或没有匹配条目时扫描分类目录；
定位候选页面后，仍需打开原始页面确认内容，不能只根据索引回答。

**业务领域知识：** [PCB 裸板 AVI/AOI](business-domain/pcb-avi-aoi.md) 供数据理解、
清洗、合成和标注按需引用；读取文档不触发任何 skill 的执行流程。
skill 通过文件读取工具使用 `knowledge/business-domain/pcb-avi-aoi.md`：
`knowledge/` 是运行时映射到已配置知识库根目录的只读逻辑路径，不相对于 skill
或会话目录。Agent 无需查询服务端环境变量或拼接仓库路径；该别名也不是终端路径。

**平台使用文档（内嵌）：**

- `basic/` — 基础用法
- `sdk/` — SDK 文档
- `studio/` — Studio 文档
- `jupyterlab/` — JupyterLab 文档

## Docker 构建

根目录 `Dockerfile` 会直接 `COPY knowledge ./knowledge`，并校验 `basic/`、`jupyterlab/`、`nodes/`、`sdk/`、`studio/` 与 `dataset_processing_workflow.py` 存在：

```bash
docker build -t pyromind-agent-server .
```
