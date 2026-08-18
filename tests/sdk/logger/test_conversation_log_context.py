import logging

import pytest

from openhands.sdk.logger import (
    conversation_log_context,
    reset_conversation_log_context,
    set_conversation_log_context,
)
from openhands.sdk.logger.logger import _install_conversation_aware_records


@pytest.fixture(autouse=True)
def _ensure_conversation_aware_records():
    _install_conversation_aware_records()
    yield


def _make_record(message: str, args: tuple = ()) -> logging.LogRecord:
    factory = logging.getLogRecordFactory()
    return factory(
        "openhands.sdk.logger.tests", logging.INFO, __file__, 1, message, args, None
    )


def test_records_prefixed_with_conversation_id(
    caplog: pytest.LogCaptureFixture,
):
    logger = logging.getLogger("openhands.sdk.logger.tests")
    with conversation_log_context("conv-abc"):
        with caplog.at_level(logging.INFO):
            logger.info("inside %s", "run")
    logger.info("outside")
    assert "[cid=conv-abc] inside run" in caplog.text
    assert "[cid=conv-abc] outside" not in caplog.text


def test_no_context_leaves_records_unprefixed(caplog: pytest.LogCaptureFixture):
    logger = logging.getLogger("openhands.sdk.logger.tests")
    with caplog.at_level(logging.INFO):
        logger.info("plain message")
    assert "plain message" in caplog.text
    assert "[cid=" not in caplog.text


def test_record_factory_sets_conversation_id_attribute():
    record = _make_record("hello %s", ("world",))
    assert record.__dict__["conversation_id"] == "-"
    with conversation_log_context("conv-9"):
        record = _make_record("hello %s", ("world",))
    assert record.msg == "[cid=conv-9] hello world"
    assert record.args == ()
    assert record.__dict__["conversation_id"] == "conv-9"


def test_set_and_reset_conversation_log_context():
    with conversation_log_context("first"):
        token = set_conversation_log_context("second")
        record = _make_record("nested")
        assert record.msg == "[cid=second] nested"
        reset_conversation_log_context(token)
        record2 = _make_record("outer again")
        assert record2.msg == "[cid=first] outer again"
