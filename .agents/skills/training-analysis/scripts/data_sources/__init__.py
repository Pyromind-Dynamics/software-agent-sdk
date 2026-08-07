"""数据源适配器包:注册表 + 已接入的数据源。

新增数据源:在 data_sources/ 下新建 <name>.py,实现与 WandbDataSource 同名方法
(extract_creds / extract_run_ref / connect / resolve_entity / probe /
get_run_data / list_runs),然后在本文件导入并注册。
"""

from __future__ import annotations

from .base import RunData, create_data_source, detect_data_source, register_data_source
from .wandb import WandbDataSource


register_data_source("wandb", WandbDataSource)

__all__ = [
    "RunData",
    "create_data_source",
    "detect_data_source",
    "register_data_source",
]
