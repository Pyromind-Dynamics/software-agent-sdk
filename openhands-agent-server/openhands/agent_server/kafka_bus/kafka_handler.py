from abc import ABC, abstractmethod

from openhands.agent_server.kafka_bus.message_event import MessageEvent


class MessageHandler(ABC):
    """
    消息处理器抽象基类

    每个 Handler 订阅一个 topic，收到消息后业务方自己处理消息体结构。
    MessageBus 按 topic 路由到对应的 Handler。
    """

    @abstractmethod
    def topic(self) -> str:
        """返回本处理器订阅的 topic"""
        raise NotImplementedError

    def concurrency(self) -> int:
        """消费并发数，默认 3。子类可覆写自定义。"""
        return 3

    @abstractmethod
    async def handle(self, event: MessageEvent):
        """处理消息（已反序列化为 MessageEvent）"""
        raise NotImplementedError

    @abstractmethod
    def group_id(self) -> str:
        """消费者组 ID，默认与 topic 同名。子类可覆写自定义。"""
        return "openhands_agent"
