"""Tests for skill trigger keyword matching precision."""

from openhands.sdk.skills import KeywordTrigger, Skill, TaskTrigger


def _keyword_skill(keywords: list[str]) -> Skill:
    return Skill(
        name="trigger-test",
        content="content",
        source="trigger-test.md",
        trigger=KeywordTrigger(keywords=keywords),
    )


def test_word_keyword_matches_whole_word():
    skill = _keyword_skill(["test"])
    assert skill.match_trigger("please test the workflow") == "test"
    assert skill.match_trigger("TEST ME") == "test"


def test_word_keyword_does_not_match_inside_longer_word():
    skill = _keyword_skill(["test"])
    # Regression: /testdata_3.csv 数据处理 must not fire the "test" trigger.
    assert skill.match_trigger("/testdata_3.csv 数据处理，过滤恶意数据") is None
    assert skill.match_trigger("the latest testing framework") is None
    assert skill.match_trigger("contesting the result") is None
    assert skill.match_trigger("unit tests were run") is None


def test_debug_workflow_scenario():
    skill = _keyword_skill(["debug", "test", "测试", "试跑"])
    assert skill.match_trigger("/testdata_3.csv 数据处理，过滤恶意数据") is None
    assert skill.match_trigger("测试这个工作流") == "测试"
    assert skill.match_trigger("试跑这个 workflow") == "试跑"
    assert skill.match_trigger("debug the workflow") == "debug"
    assert skill.match_trigger("帮我 debug 一下") == "debug"


def test_non_word_keywords_keep_substring_semantics():
    skill = _keyword_skill(["/review", "best practices", "hidden-tag"])
    assert skill.match_trigger("please /review the pr") == "/review"
    assert skill.match_trigger("follow best practices here") == "best practices"
    assert skill.match_trigger("set hidden-tag to on") == "hidden-tag"


def test_empty_keyword_never_matches():
    skill = _keyword_skill([""])
    assert skill.match_trigger("anything at all") is None


def test_task_trigger_uses_word_boundaries_too():
    skill = Skill(
        name="task-skill",
        content="content",
        source="task.md",
        trigger=TaskTrigger(triggers=["task"]),
    )
    assert skill.match_trigger("do the task now") == "task"
    assert skill.match_trigger("multi-tasking is hard") is None
