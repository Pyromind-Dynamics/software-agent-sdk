# PyroMind Knowledge Base

Agent 生产运行时通过 `PYROMIND_KNOWLEDGE_BASE_PATH`（默认指向本目录）检索文档。

## 目录结构

| 路径 | 内容 |
|------|------|
| [index.md](index.md) | 唯一的逐页文档大纲；按标题、摘要和标签定位候选页面 |
| `basic/`、`sdk/`、`studio/`、`jupyterlab/` | 内嵌平台文档（Agent 运行时检索根目录） |
| `nodes/<NodeType>/<NodeType>.md` | 节点 I/O、参数与端口定义 |
| `business-domain/` | 跨 skill 共用的业务领域知识，如 PCB 裸板 AVI/AOI 数据与判别参考 |
| `dataset_processing_workflow.py` | 工作流 DSL 样例 |

[PyroMind Python SDK](https://pypi.org/project/pyromind-sdk/) 通过 PyPI 安装，`make build`（`uv sync --dev`）会默认安装。

## 节点契约来源

`nodes/` 的输入输出表按 2026-11-10 生产环境 `us-west-1` 的
`GET /user_node/all_with_status` 返回定义同步。该目录包含系统节点和共享节点；
`GET /std2/studio_api/api/object_info` 只覆盖其中的系统节点，不能用于判断整个节点库是否完整。

表中的必填状态与默认值分别记录：有默认值不表示输入可省略；“未声明”表示 schema 未提供默认值，
不代表实际运行时一定没有默认行为。GPU、中心仓模型和数据集候选随集群变化；
其他环境以该环境的实时定义与 `validate_workflow_dsl` 校验为准。
schema 中的候选列表不证明数据或模型当前已成功加载，运行能力仍需实际任务验证。

`dataset_processing_workflow.py` 保留数据处理拓扑示例；2026-11-10 的 us-west-1 校验中，
PathJoin→DatasetToJsonl、DatasetToJsonl→DatasetConfig 两条连线存在类型不匹配。
不要将该文件直接作为已通过校验的模板，也不要仅为绕过校验插入未经验证的类型转换。

## Agent 检索路径

Agent 需要查知识库时，先用文件读取工具打开 `knowledge/index.md`，再打开匹配的原始页面；
仅在索引缺失或没有匹配条目时扫描分类目录，不能只根据索引回答。

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
