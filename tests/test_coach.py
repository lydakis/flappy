from llm.coach import Coach


class StubClient:
    def __init__(self, response: str) -> None:
        self._response = response

    def invoke_text(self, messages, *, metadata=None):
        return self._response


def test_parse_directive_basic():
    client = StubClient("SUBGOAL: click submit\nMASK_ALLOW: 0;1")
    coach = Coach(client)
    directive = coach._parse_response(client._response)
    assert directive.subgoal == "click submit"
    assert directive.mask_allow == ["0", "1"]
