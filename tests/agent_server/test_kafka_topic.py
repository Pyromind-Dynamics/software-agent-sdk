from openhands.agent_server.kafka_bus.kafka_bus import KafkaMessageBus
from openhands.agent_server.kafka_bus.kafka_topic import KafkaTopic
from openhands.sdk.utils import env_util


def test_kafka_env_can_target_pre_while_server_stays_in_dev(monkeypatch):
    monkeypatch.setenv("APP_ENV", "dev")
    monkeypatch.setenv("KAFKA_ENV", "pre")

    assert env_util.is_dev()
    assert not env_util.is_kafka_dev()
    assert KafkaTopic.WORKFLOW_MONITOR.resolve() == "workflow_monitor_topic_pre"
    assert not KafkaMessageBus()._is_dev()


def test_kafka_env_falls_back_to_server_environment(monkeypatch):
    monkeypatch.setenv("APP_ENV", "pre2")
    monkeypatch.delenv("KAFKA_ENV", raising=False)

    assert env_util.get_kafka_env_value() == "pre2"
    assert KafkaTopic.WORKFLOW_MONITOR.resolve() == "workflow_monitor_topic_pre2"
