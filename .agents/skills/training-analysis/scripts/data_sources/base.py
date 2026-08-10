"""数据源抽象层:RunData 标准格式 + 注册表。

新增数据源只需两步(见 SKILL.md 技术说明):
1. 新建 ``data_sources/<name>.py``,按 ``WandbDataSource`` 骨架实现同名方法
   (鸭子类型,无需继承)
2. 在 ``__init__.py`` 导入该类并 ``register_data_source("<name>", Cls)``

数据分析层(``analysis_helpers`` / CLI 分析命令)只消费 ``RunData``,无需改动。
"""

from __future__ import annotations

from typing import Any, TypedDict


class RunData(TypedDict):
    """数据源无关的 run 标准数据格式(所有分析函数的唯一输入)。"""

    run_id: str
    display_name: str
    state: str
    config: dict[str, Any]
    summary: dict[str, Any]
    history: list[dict[str, Any]]


# 注册表: name -> DataSource 类(鸭子类型,不强制继承)
_SOURCES: dict[str, type] = {}


def register_data_source(name: str, cls: type) -> None:
    """登记一个数据源适配器类。"""
    _SOURCES[name] = cls


def detect_data_source(nodes: list[dict[str, Any]]) -> str | None:
    """自动探测数据源:遍历注册表,第一个 extract_creds 非空的命中。

    判据即适配器自身的 ``extract_creds(nodes)`` 是否识别到本数据源凭证,
    新增数据源无需改动探测逻辑。
    """
    for name, cls in _SOURCES.items():
        if cls().extract_creds(nodes):
            return name
    return None


def create_data_source(name: str) -> Any:
    """按名称创建数据源实例;未知名称抛 ValueError。"""
    if name not in _SOURCES:
        raise ValueError(
            f"unknown data source: {name} (registered: {sorted(_SOURCES)})"
        )
    return _SOURCES[name]()
