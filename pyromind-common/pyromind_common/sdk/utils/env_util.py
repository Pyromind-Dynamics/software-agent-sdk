import os


def is_dev() -> bool:
    mock = get_env_value()
    return mock == "dev"


def get_env_value() -> str:
    app_env: str = os.getenv("APP_ENV", "dev")
    return app_env


def is_prod_or_pre() -> bool:
    env = get_env_value()
    return env in ["pre", "prod", "pre2"]


def is_production() -> bool:
    """
    判断是否是生产环境
    """
    env = get_env_value()
    return env == "prod"


def is_pre() -> bool:
    """
    是否是预发环境
    """
    env = get_env_value()
    return env in ["pre", "pre2"]


def get_cluster_region():
    """
    获取当前集群配置信息
    由于当前项目统一部署，因此不会有集群配置信息，为了实现集群隔离，这里返回 "normal"
    """
    return "normal"


def get_pod_name():
    pod_name = os.getenv("POD_NAME", "dev_pod_name")
    return pod_name


# ===========================================================================
# Kafka 配置
# ===========================================================================


def get_kafka_env_value() -> str:
    """Return the Kafka environment without changing the server runtime mode."""
    return os.getenv("KAFKA_ENV", get_env_value())


def is_kafka_dev() -> bool:
    return get_kafka_env_value() == "dev"


def is_kafka_pre() -> bool:
    return get_kafka_env_value() in {"pre", "pre2"}


def get_kafka_brokers() -> str | None:
    """获取当前集群的 Kafka broker 地址"""
    return os.getenv(
        "KAFKA_URL",
        "kafka-controller-0.kafka-controller-headless.kafka.svc.cluster.local:9092,kafka-controller-1.kafka-controller-headless.kafka.svc.cluster.local:9092,kafka-controller-2.kafka-controller-headless.kafka.svc.cluster.local:9092",
    )
