from concurrent.futures import ThreadPoolExecutor

from openhands.sdk import LocalConversation
from openhands.sdk.agent import Agent
from openhands.sdk.conversation.state import ActiveLongTask
from openhands.sdk.event.llm_convertible import MessageEvent
from openhands.sdk.testing import TestLLM


def _conversation(tmp_path) -> LocalConversation:
    llm = TestLLM.from_messages([], model="default-model", usage_id="test-llm")
    agent = Agent(llm=llm, tools=[], include_default_tools=[])
    return LocalConversation(agent=agent, workspace=tmp_path, visualizer=None)


def test_register_and_remove_from_worker_thread_while_step_holds_lock(tmp_path):
    """Regression for the #3485 lock pattern: while run()/arun() holds the
    state lock across an agent step, tools and callbacks on worker threads
    must not re-acquire it. Both the tool-side registration and the
    callback-side removal used to deadlock against the run loop — the
    conversation stayed ``running`` and the terminal callback appeared to
    be consumed without any effect.
    """
    conv = _conversation(tmp_path)
    task = ActiveLongTask(task_id="t1", kind="data_cleaning", status="Pending")

    def worker():
        conv.register_active_long_task(task)
        conv.send_agent_message("已提交任务 t1")
        return conv.remove_active_long_task("t1")

    with conv._state:  # Simulates the run loop holding the lock across astep().
        conv._step_holds_state_lock = True
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(worker)
            # A regression (re-acquiring the lock on the worker thread)
            # surfaces as a timeout instead of a hung suite.
            removed = future.result(timeout=5)

    assert removed == task
    assert conv.state.active_long_tasks == []
    agent_events = [
        event
        for event in conv.state.events
        if isinstance(event, MessageEvent) and event.source == "agent"
    ]
    assert len(agent_events) == 1


def test_register_is_reentrant_for_the_owning_thread(tmp_path):
    """When the caller already owns the lock (e.g. a sync step on the run
    loop thread), the lock is acquired normally — FIFOLock is reentrant
    for the owning thread, so registration must not be skipped.
    """
    conv = _conversation(tmp_path)
    task = ActiveLongTask(task_id="t2", kind="data_preparation", status="Pending")

    with conv._state:
        conv._step_holds_state_lock = True
        conv.register_active_long_task(task)

    assert conv.state.active_long_tasks == [task]
