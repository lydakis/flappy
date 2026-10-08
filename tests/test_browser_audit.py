import copy

from scripts.run_browser_audit import canonical_dom


def test_browser_observation_canonicalizes_only_opaque_ids():
    dom = {
        "strings": ["uuid-one", "Select Alpha", "Alpha"],
        "documents": [
            {"frameId": 0, "nodes": {"backendNodeId": [37, 91], "nodeValue": [1, 2]}}
        ],
    }
    other = copy.deepcopy(dom)
    other["strings"][0] = "uuid-two"
    other["documents"][0]["nodes"]["backendNodeId"] = [50, 75]
    expected = canonical_dom(dom)
    assert expected == canonical_dom(other)
    assert expected["strings"][1:] == dom["strings"][1:]
    assert expected["documents"][0]["nodes"]["nodeValue"] == [1, 2]
    assert dom["strings"][0] == "uuid-one"
