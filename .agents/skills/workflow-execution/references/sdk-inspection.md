# SDK 查询任务状态与节点日志

系统启用 SDK 时，在 terminal 使用 `"$PYROMIND_SDK_PYTHON"` 和当前账户环境。
`inspect_workflow_task` Tool 已停止注入；查询使用下面的普通 Python 函数。
SDK 关闭且实际清单仍列出旧 Tool 时，按其参数查询。

## 查询任务和节点

```python
import sys

from pyromind_sdk import PyroMindAPIClient
from pyromind_common.platform.inspection import inspect_workflow_task

task_id = sys.argv[1]
with PyroMindAPIClient(max_retries=0) as client:
    result = inspect_workflow_task(client, task_id)
    print(result.model_dump_json())
```

这是随 `pyromind-platform-helpers` 安装的普通函数，通过 SDK 查询平台。
只安装原生 SDK 不等于已经安装该模块；执行环境缺失时先修复安装，不导入 Host 源码。
使用本次提交返回的真实 task ID，不重新创建任务来检查状态。

返回内容包括任务原始状态 `task_status`、归一化状态 `task_state`、节点列表 `nodes`、
失败节点 `failed_nodes`、可选节点 `selected_node` 和日志。节点带 `node_id`、可选
`node_code`、名称、类型及 `raw_status` / `state`。

SDK 0.1.17 的 `get_task()` 节点模型没有 `status`，解析时会丢弃该字段。该函数通过
`client.studio.get()` 读取原始任务响应，保留平台实际返回的节点状态。
平台未返回的状态保持 `unknown`，并通过 `node_status_available=false` 表明状态信息
不完整。不得用开始时间、结束时间、任务整体状态或退出码补造节点状态。
`node_status_available` 只表示原始状态字段是否齐全；平台自定义状态仍可能归一化为
`unknown`。`failed_nodes=[]` 不能证明所有节点成功，尤其在状态缺失或存在 unknown 时。

## 查询指定节点的日志尾部

先从返回的节点列表选择真实 `node_id`，再读取该节点日志。不要将 DSL 节点顺序当作
平台节点数据库 ID；存在 `node_code` 时也优先使用返回的 `node_id`。

```python
node_id = sys.argv[2]
with PyroMindAPIClient(max_retries=0) as client:
    result = inspect_workflow_task(
        client,
        task_id,
        node_id=node_id,
        include_log=True,
        tail_lines=100,
        max_log_chars=20000,
    )
    print(result.model_dump_json())
```

`include_log=True` 必须同时提供 `node_id`。默认返回最后 100 行、最多 20000 字符，
上限分别为 1000 行、100000 字符。截断发生在 SDK 取得日志之后，不代表网络响应只下载
这些行。`node_log_available=false` 时检查 `log_error`；日志不可用不应抹掉已取得的任务
和节点状态，也不能据此判断节点成功。查询任务失败会抛出 `WorkflowInspectionError`。

需要按时间向前分段读日志时，使用原生 SDK：

```python
with PyroMindAPIClient(max_retries=0) as client:
    lines = client.studio.scroll_node_logs(task_id, node_id, expect_count=500)
    print("\n".join(lines))
```

这是前向分段接口，不是读取日志末尾；使用实际观察到的时间戳续读，不伪造游标。
按需检查一次后返回结果，待终态回调继续处理，不在 terminal 循环轮询等待。
返回日志只保留诊断所需片段，不保存或回显凭据和完整请求响应。

失败时对照本次实际提交图、绑定数据和对应节点日志定位原因。任务成功后仍按用户要求
核验产物，再进入训练分析或评测；平台状态本身不证明业务产物已通过验收。
