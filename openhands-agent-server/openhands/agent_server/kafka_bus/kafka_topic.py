"""
Kafka Topic 枚举 — 统一管理所有 Kafka Topic。

每个 Topic 包含三个属性:
  - base_name: 基础名称，resolve() 时自动根据环境和集群拼接
  - cluster_scoped: 是否需要集群级 topic 隔离
      True  → topic 名称包含集群标识，每个集群独立 topic
      False → 所有集群共享同一 topic
  - concurrency: 消费并发数（默认 3），可在枚举定义时按 topic 自定义

命名规则:
  cluster_scoped=False:
    prod:  {base_name}
    pre:   {base_name}_pre
    pre2:  {base_name}_pre2

  cluster_scoped=True:
    prod:  {base_name}_{cluster_region}
    pre:   {base_name}_pre_{cluster_region}
    pre2:  {base_name}_pre2_{cluster_region}

使用方式:
  from app.common.kafka_topic import KafkaTopic
  topic = KafkaTopic.TRAINING_BILLING.resolve()  # → "training_billing_topic_us-west-1"
  KafkaTopic.TRAINING_BILLING.cluster_scoped      # → True
"""

from enum import Enum

from openhands.sdk.utils import env_util


class KafkaTopic(Enum):
    """
    Kafka Topic 枚举。

    value 格式: (base_name, cluster_scoped, concurrency)
      - base_name:      topic 基础名称（resolve 时按环境+集群拼接）
      - cluster_scoped: 是否需要集群级 topic 隔离
      - concurrency:    消费并发数（默认 3）
    """

    WORKFLOW_MONITOR = ("workflow_monitor_topic", False, 3)  # 工作流监控
    DLQ = ("dead_letter_queue_topic", False, 3)  # 死信队列（所有消费失败消息的中转站）

    def __init__(self, base_name: str, cluster_scoped: bool, concurrency: int = 3):
        self._base_name = base_name
        self._cluster_scoped = cluster_scoped
        self._concurrency = concurrency

    @property
    def base_name(self) -> str:
        """Topic 基础名称"""
        return self._base_name

    @property
    def cluster_scoped(self) -> bool:
        """是否需要集群级 topic 隔离：True=每集群独立 topic，False=集群共享"""
        return self._cluster_scoped

    @property
    def concurrency(self) -> int:
        """消费并发数"""
        return self._concurrency

    def resolve(self) -> str:
        """
        解析为实际 topic 名称。

        cluster_scoped=False:
          prod → "user_socket_notify_topic"
          pre  → "user_socket_notify_topic_pre"

        cluster_scoped=True:
          prod → "training_billing_topic_us-west-1"
          pre  → "training_billing_topic_pre_us-west-1"
        """
        env = env_util.get_kafka_env_value()
        cluster_region = env_util.get_cluster_region()

        if self.cluster_scoped:
            # 集群隔离: {base_name}[_{env}]_{cluster_region}
            if env_util.is_kafka_pre():
                return f"{self.base_name}_{env}_{cluster_region}"
            return f"{self.base_name}_{cluster_region}"
        else:
            # 集群共享: {base_name}[_{env}]
            if env_util.is_kafka_pre():
                return f"{self.base_name}_{env}"
            return self.base_name
