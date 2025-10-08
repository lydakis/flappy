from envs.browsergym_client import BrowserGymEnvWrapper


def test_tool_schema_contains_click():
    schemas = BrowserGymEnvWrapper.tool_schemas()
    assert "click" in schemas
    assert "selector" in schemas["click"]["properties"]
