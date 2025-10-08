from llm import prompts


def test_coach_prompts_have_placeholders():
    assert "Coach" in prompts.COACH_SYSTEM_PROMPT
    assert "{dom}" in prompts.COACH_DEVELOPER_PROMPT
    assert "{task_id}" in prompts.REFLECTION_PROMPT
