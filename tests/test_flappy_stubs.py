from flappy import dsl, macro_registry, memory, rag, synth, verify


def test_dsl_render_roundtrip():
    node = dsl.make_leaf(dsl.DSLVerb.CLICK, "#submit")
    parsed = dsl.DSLNode.from_dict(node.to_dict())
    assert parsed.verb == node.verb
    assert parsed.args == node.args
    assert "Click" in dsl.render_plan(parsed)


def test_macro_registry_stats():
    registry = macro_registry.MacroRegistry()
    macro = macro_registry.Macro(name="HandleCookie", body=[dsl.make_leaf(dsl.DSLVerb.CLICK, "button")])
    registry.propose(macro)
    registry.record_success("HandleCookie", success=True)
    stats = registry.stats()
    assert stats["HandleCookie"]["usage"] == 1
    assert stats["HandleCookie"]["success"] == 1


def test_note_store_roundtrip(tmp_path):
    store = memory.NoteStore(tmp_path / "notes.jsonl")
    note = memory.Note(id="n1", url="https://example.com", title="Title", snippet="Snippet")
    store.append(note)
    loaded = store.load()
    assert loaded and loaded[0].id == "n1"
    results = store.search("title")
    assert results


def test_rag_stub():
    note = memory.Note(id="n1", url="https://example.com", title="T", snippet="Answer")
    answer = rag.SimpleRAG().answer("Q?", [note])
    assert "Answer" in answer.text
    assert answer.citations == ["https://example.com"]


def test_plan_synthesiser_identity():
    leaf = dsl.make_leaf(dsl.DSLVerb.CLICK, "#submit")
    sketch = synth.Sketch(root=leaf, holes=[])
    context = {"selectors": ["#submit"]}
    proposals = list(synth.PlanSynthesiser().enumerate(sketch, context=context))
    assert proposals
    assert proposals[0].root.to_dict() == leaf.to_dict()


def test_plan_synthesiser_empty_selectors_rejects_missing():
    leaf = dsl.make_leaf(dsl.DSLVerb.CLICK, "#missing")
    sketch = synth.Sketch(root=leaf, holes=[])
    proposals = list(
        synth.PlanSynthesiser().enumerate(sketch, context={"selectors": []})
    )
    assert not proposals


def test_verifier_stub():
    verifier = verify.PlanVerifier()
    result = verifier.verify(
        dsl.make_leaf(dsl.DSLVerb.CLICK, "#submit"),
        trace=[],
        context={"selectors": ["#submit"]},
    )
    assert result.ok
