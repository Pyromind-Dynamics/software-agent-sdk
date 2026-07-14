"""
Kafka 消息总线 — 统一管理 Producer/Consumer 生命周期

提供:
  - KafkaMessageBus.send             生产者（复用长连接）
  - KafkaMessageBus.register_handler 注册消息处理器
  - KafkaMessageBus.start_consumer / stop  消费者生命周期

多集群支持:
  - cluster_scoped topic: topic 名称包含集群标识，每集群独立 topic
  - 非 cluster_scoped topic: 所有集群共享同一 topic
"""

import asyncio
import json
import uuid
from datetime import datetime

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

from openhands.agent_server.kafka_bus.kafka_handler import MessageHandler
from openhands.agent_server.kafka_bus.kafka_topic import KafkaTopic
from openhands.agent_server.kafka_bus.message_event import MessageEvent
from openhands.sdk.logger import get_logger
from openhands.sdk.utils import env_util


logger = get_logger(__name__)


# ──────────────────────────────────────────────
# KafkaMessageBus
# ──────────────────────────────────────────────


class KafkaMessageBus:
    """
    统一管理 Kafka Producer（发送）和 Consumer（消费）。

    按 topic 路由消息到已注册的 MessageHandler。
    支持 cluster_scoped 和全局两种消费模式。
    """

    def __init__(self):
        self._producer: AIOKafkaProducer | None = None
        self._consumers: list[AIOKafkaConsumer] = []
        self._consumer_tasks: list[asyncio.Task] = []
        self._restart_tasks: set[asyncio.Task] = set()
        self._pending_handlers: set[asyncio.Task] = set()
        self._handlers: dict[str, MessageHandler] = {}  # resolved_topic -> handler
        self._startup_task: asyncio.Task | None = None

    # ── Producer ──

    async def _get_or_create_producer(self) -> AIOKafkaProducer:
        if self._producer is None:
            brokers = env_util.get_kafka_brokers()
            if not brokers:
                raise RuntimeError(
                    "[MessageBus] KAFKA_BROKERS 未配置，无法创建 Producer"
                )
            self._producer = AIOKafkaProducer(
                bootstrap_servers=brokers,
                request_timeout_ms=40000,
            )
            await self._producer.start()
        return self._producer

    async def send(self, topic: KafkaTopic, message: MessageEvent) -> bool:
        """
        发送消息到指定 topic，复用长连接 Producer。

        Args:
            topic: Kafka Topic 枚举（自动根据环境 resolve 实际 topic 名称）
            message: MessageEvent 消息对象（自动序列化为 JSON，自动填充 message_id）

        Returns:
            是否发送成功
        """
        # 自动填充 message_id（UUID v4），用于消息唯一标识与全链路追踪
        if not message.message_id:
            message.message_id = str(uuid.uuid4())

        # 自动填充 source_app
        message.source_app = "openhands-agent"

        # 自动填充集群标识，用于消息追踪和日志
        message.cluster = env_util.get_cluster_region()
        message.env = env_util.get_kafka_env_value()
        message.topic = topic.resolve()
        message.timestamp = str(_current_timestamp())

        resolved_topic = topic.resolve()
        message_json = message.to_json()
        try:
            producer = await self._get_or_create_producer()
            await producer.send_and_wait(
                topic=resolved_topic,
                value=message_json.encode("utf-8"),
            )
            logger.info(
                f"[MessageBus] 已发送: topic={resolved_topic}, "
                f"message_id={message.message_id},biz_code={message.biz_code}"
            )
            return True
        except Exception as e:
            logger.error(
                "[MessageBus] 发送失败: message_id=%s, biz_code=%s, error=%s",
                message.message_id,
                message.biz_code,
                e,
            )
            return False

    # ── Handler 注册 ──

    def register_handler(self, handler: MessageHandler):
        """注册一个消息处理器，按 topic 路由"""
        t = handler.topic()
        self._handlers[t] = handler
        logger.info(f"[MessageBus] 注册 handler, topic={t}")

    # ── Consumer ──

    async def start_consumer(self):
        """
        注册所有 handler，后台异步启动 Consumer（不阻塞 lifespan）。

        Consumer 启动失败会在后台自动重试，直到成功或应用关闭。
        """
        from openhands.agent_server.kafka_bus.handlers import ALL_HANDLERS

        if self._is_dev():
            logger.warning("[MessageBus] 开发环境，不启动 Consumer")
            return

        for h in ALL_HANDLERS:
            self.register_handler(h)

        brokers = env_util.get_kafka_brokers()
        if not brokers:
            logger.warning("[MessageBus] KAFKA_BROKERS 未配置，无法启动 Consumer")
            return

        if not self._handlers:
            logger.warning("[MessageBus] 无注册 handler，跳过 Consumer 启动")
            return

        self._startup_task = asyncio.create_task(self._background_start_consumers())

    async def _background_start_consumers(self):
        """后台重试启动所有 Consumer，失败后每 30s 重试，直到全部成功"""
        RETRY_INTERVAL = 30

        pending = list(self._handlers.items())
        attempt = 0

        while pending:
            attempt += 1
            failed = []

            for resolved_topic, handler in pending:
                group_id = handler.group_id()
                ok = await self._create_consumer(
                    resolved_topic, group_id, handler.concurrency()
                )
                if ok:
                    logger.info(
                        "[MessageBus] Consumer 已启动: topic=%s, group=%s, "
                        "concurrency=%s",
                        resolved_topic,
                        group_id,
                        handler.concurrency(),
                    )
                else:
                    failed.append((resolved_topic, handler))

            if not failed:
                logger.info("[MessageBus] 所有 Consumer 启动成功")
                return

            pending = failed
            logger.error(
                "[MessageBus] %s 个 Consumer 启动失败，%ss 后重试（第 %s 轮）",
                len(failed),
                RETRY_INTERVAL,
                attempt,
            )
            await asyncio.sleep(RETRY_INTERVAL)

    async def _create_consumer(
        self, topic: str, group_id: str, concurrency: int = 3
    ) -> bool:
        """创建并启动一个 AIOKafkaConsumer，返回是否成功"""
        brokers = env_util.get_kafka_brokers()
        if not brokers:
            logger.error("[MessageBus] KAFKA_BROKERS 未配置，无法创建 Consumer")
            return False
        consumer = AIOKafkaConsumer(
            topic,
            bootstrap_servers=brokers,
            group_id=group_id,
            auto_offset_reset="latest",
            enable_auto_commit=True,
            auto_commit_interval_ms=5000,
            request_timeout_ms=40000,
            max_poll_interval_ms=900000,
            session_timeout_ms=30000,
            heartbeat_interval_ms=5000,
        )
        try:
            await consumer.start()
            self._consumers.append(consumer)
            task = asyncio.create_task(
                self._consume_loop(consumer, topic, group_id, concurrency)
            )
            self._consumer_tasks.append(task)
            return True
        except Exception as e:
            logger.error(
                "[MessageBus] Consumer 启动失败: topic=%s, group=%s, error=%s",
                topic,
                group_id,
                e,
            )
            try:
                await consumer.stop()
            except Exception:
                pass
            return False

    async def _consume_loop(
        self,
        consumer: AIOKafkaConsumer,
        topic: str,
        group_id: str,
        concurrency: int = 3,
    ):
        """Serial consume loop; offset auto-commit. Rebuild consumer on loop failure.

        ``concurrency`` is kept for API compatibility with handler registration
        but messages are handled one-at-a-time in this coroutine.
        """
        try:
            async for msg in consumer:
                try:
                    if msg.value is None:
                        logger.error(
                            f"[MessageBus] 空消息体: topic={msg.topic}, "
                            f"offset={msg.offset}"
                        )
                        continue
                    payload = json.loads(msg.value.decode("utf-8"))
                    event = MessageEvent.from_dict(payload)
                    event.topic = msg.topic
                except Exception:
                    raw = msg.value[:200] if msg.value is not None else None
                    logger.error(
                        f"[MessageBus] 消息反序列化失败，跳过: "
                        f"topic={getattr(msg, 'topic', topic)}, "
                        f"offset={getattr(msg, 'offset', None)}, value={raw!r}",
                        exc_info=True,
                    )
                    continue

                await self._handle_one(event)
        except Exception as e:
            logger.error(
                f"[MessageBus] 消费循环异常退出: topic={topic}, error={e}",
                exc_info=True,
            )
            logger.info(f"[MessageBus] 触发 Consumer 自动重建: topic={topic}")
            restart_task = asyncio.create_task(
                self._restart_consumer(consumer, topic, group_id, concurrency)
            )
            self._restart_tasks.add(restart_task)
            restart_task.add_done_callback(self._restart_tasks.discard)
        finally:
            logger.info(f"[MessageBus] 消费循环结束: topic={topic}")

    def _is_dev(self) -> bool:
        return env_util.is_kafka_dev()

    async def _restart_consumer(
        self,
        old_consumer: AIOKafkaConsumer,
        topic: str,
        group_id: str,
        concurrency: int,
    ):
        """Consumer 异常退出后后台重建，每 30s 重试直到成功"""
        RETRY_INTERVAL = 30
        attempt = 0

        if self._is_dev():
            logger.warning("[MessageBus] 开发环境，不启动 Consumer")
            return

        if old_consumer in self._consumers:
            self._consumers.remove(old_consumer)

        try:
            await old_consumer.stop()
        except Exception as e:
            logger.warning(
                f"[MessageBus] 停止旧 Consumer 失败: topic={topic}, error={e}"
            )

        while True:
            attempt += 1
            if await self._create_consumer(topic, group_id, concurrency):
                logger.info(f"[MessageBus] Consumer 重建成功: topic={topic}")
                return
            logger.error(
                f"[MessageBus] Consumer 重建失败: topic={topic}（第 {attempt} 次），"
                f"{RETRY_INTERVAL}s 后重试"
            )
            await asyncio.sleep(RETRY_INTERVAL)

    async def _handle_one(self, event: MessageEvent):
        """处理单条消息：消费级重试，失败发回原 topic（attempt+1），>=3 次走 DLQ"""
        message_id = event.message_id
        attempt = event.attempt or 0
        logger.info(
            f"[MessageBus] 收到消息: message_id={message_id}, attempt={attempt}"
        )

        topic = event.topic
        if not topic:
            logger.warning("[MessageBus] 消息缺少 topic")
            return
        handler = self._handlers.get(topic)
        if not handler:
            logger.warning("[MessageBus] 无 handler 处理: topic=%s", topic)
            return

        group_id = handler.group_id()
        try:
            await handler.handle(event)
        except Exception as e:
            attempt += 1
            if attempt >= 3:
                error_msg = f"消费重试 {attempt} 次仍失败: {e}"
                event.error_message = error_msg
                logger.error(
                    "[MessageBus] 重试耗尽: message_id=%s, topic=%s, attempt=%s",
                    message_id,
                    event.topic,
                    attempt,
                )
                await self._send_to_dlq(event, group_id)
            else:
                event.attempt = attempt
                event.error_message = str(e)
                logger.warning(
                    f"[MessageBus] 消费失败，发回原 topic 重试: topic={event.topic}, "
                    f"message_id={message_id}, biz_code={event.biz_code}, "
                    f"attempt={attempt}, warning={e}"
                )
                try:
                    producer = await self._get_or_create_producer()
                    await producer.send_and_wait(
                        topic=event.topic, value=event.to_json().encode("utf-8")
                    )
                except Exception as resend_err:
                    logger.error(f"[MessageBus] 重发失败: {resend_err}", exc_info=True)
                    await self._send_to_dlq(event, group_id)

    async def _send_to_dlq(self, event: MessageEvent, group: str) -> bool:
        """将失败消息投递到死信 topic（委托 DLQService 处理）"""
        # Lazy import avoids circular import with dlq_service → kafka_message_bus.
        from openhands.agent_server.kafka_bus.dlq import dlq_service

        return await dlq_service.send_to_dlq(event=event, group=group)

    async def stop(self):
        """优雅关闭：取消后台启动 → 停止消费循环 → 等待 in-flight handler → 关闭连接"""
        # 取消后台 Consumer 启动 task
        if self._startup_task and not self._startup_task.done():
            self._startup_task.cancel()
            try:
                await self._startup_task
            except (asyncio.CancelledError, Exception):
                pass
            self._startup_task = None

        # 取消所有消费循环和重建 task
        for task in self._consumer_tasks:
            task.cancel()
        for task in self._restart_tasks:
            task.cancel()
        all_tasks = self._consumer_tasks + list(self._restart_tasks)
        if all_tasks:
            await asyncio.gather(*all_tasks, return_exceptions=True)
        self._consumer_tasks.clear()
        self._restart_tasks.clear()

        # 等待正在执行中的 handler 任务完成
        if self._pending_handlers:
            await asyncio.gather(*self._pending_handlers, return_exceptions=True)
            self._pending_handlers.clear()

        # 关闭所有 consumer
        for consumer in self._consumers:
            try:
                await consumer.stop()
            except Exception:
                pass
        self._consumers.clear()

        if self._producer is not None:
            try:
                await self._producer.stop()
            except Exception:
                pass
            self._producer = None

        logger.info("[MessageBus] 已停止")


def _current_timestamp() -> int:
    return int(datetime.now().timestamp())


# ── 全局单例 ──
kafka_message_bus = KafkaMessageBus()
