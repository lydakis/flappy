from envs.selectors import extract_interactive_selectors


def test_extract_interactive_selectors_finds_button():
    dom = '<div><button id="submit-btn" aria-label="Submit">Submit</button></div>'
    selectors = extract_interactive_selectors(dom, max_candidates=5)
    assert any("button" in selector for selector in selectors)
