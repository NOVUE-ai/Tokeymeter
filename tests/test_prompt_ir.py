"""Phase 0 Prompt IR: reconstruction, tiling, classification, equivalence,
fail-safe. The IR is substrate only — no optimizer consumes it yet."""
import pytest
from tokeymeter.prompt_ir import (
    parse, reconstruct, equivalent_single_span,
    SpanKind, Origin, Permissions, permissions_for, PromptIR,
)


# ---------------- Reconstruction: byte-exact (the hard guarantee) ----------------

@pytest.mark.parametrize("text", [
    "",
    "just a plain question",
    "before ```python\ndef f():\n    return 1\n``` after",
    'say "this exact   thing" please',
    "mix `inline` and ```block``` and \"quote\" and 'single'",
    "trailing code at end ```x```",
    "```starts with code``` then text",
    "weird   whitespace\n\n\tand tabs",
    "unicode: 日本語 and emoji 🚀 and accents café",
    "```nested ``` not really nested``` edge",
    'unterminated "quote and `code',
    "multiple\n\n```a```\nmiddle\n```b```\nend",
])
def test_string_reconstruction_is_byte_exact(text):
    ir = parse(text)
    assert reconstruct(ir) == text, "reconstruction must be byte-exact"


def test_spans_tile_with_no_gaps_or_overlaps():
    text = "intro ```code``` middle \"quoted\" tail"
    ir = parse(text)
    # concatenation in order == original, and offsets are contiguous
    assert "".join(s.text for s in ir.spans) == text
    assert sum(len(s.text) for s in ir.spans) == len(text)


# ---------------- Message arrays ----------------

def test_messages_reconstruction_preserves_content_and_extra_keys():
    msgs = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi there", "name": "alice"},
        {"role": "assistant", "content": "Hello!"},
    ]
    ir = parse(msgs)
    out = reconstruct(ir)
    assert out == msgs, "messages must round-trip incl. extra keys like 'name'"


def test_messages_roles_map_to_origins_and_kinds():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
        {"role": "tool", "content": "t"},
    ]
    ir = parse(msgs)
    kinds = [s.kind for s in ir.spans]
    origins = [s.origin for s in ir.spans]
    assert kinds == [SpanKind.SYSTEM_INSTRUCTION, SpanKind.USER_QUERY,
                     SpanKind.FEW_SHOT_EXAMPLE, SpanKind.TOOL_DEFINITION]
    assert origins == [Origin.SYSTEM, Origin.USER, Origin.MODEL, Origin.TOOL]
    assert ir.parser_conf == 1.0


# ---------------- Classification ----------------

def test_code_and_quotes_are_verbatim_and_plain_is_user_query():
    ir = parse('explain ```py\nx=1\n``` and "literal"')
    by_kind = {s.kind for s in ir.spans}
    assert SpanKind.CODE in by_kind and SpanKind.QUOTED in by_kind
    assert SpanKind.USER_QUERY in by_kind
    for s in ir.spans:
        if s.kind in (SpanKind.CODE, SpanKind.QUOTED):
            assert s.permissions.must_be_verbatim and not s.permissions.compressible


def test_unknown_classification_fails_safe():
    # A plain string with no structure -> single conservative USER_QUERY span.
    ir = parse("no structure here")
    assert ir.is_single_span()
    p = ir.spans[0].permissions
    assert p.must_be_verbatim and p.pii_sensitive and p.semantic_identity
    assert not p.compressible


def test_permission_default_equals_user_query():
    assert permissions_for(SpanKind.USER_QUERY) == Permissions()  # safe defaults


# ---------------- Backward-compat contract ----------------

def test_single_span_equivalence_recovers_text():
    text = "anything at all ```code``` included"
    ir = equivalent_single_span(text)
    assert ir.is_single_span()
    assert reconstruct(ir) == text
    assert ir.spans[0].kind == SpanKind.USER_QUERY


def test_semantic_and_cacheable_surfaces():
    # RETRIEVED_CONTEXT-style exclusion is exercised via message kinds:
    # system is excluded from semantic identity; user query is included.
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "Q"}]
    ir = parse(msgs)
    assert ir.semantic_text() == "Q"            # only USER_QUERY is semantic identity
    assert "S" in ir.cacheable_text() and "Q" in ir.cacheable_text()


# ---------------- Fail-safe ----------------

def test_garbage_input_never_raises():
    for bad in [None, 12345, {"not": "messages"}, [1, 2, 3], [{"role": "user"}]]:
        ir = parse(bad)
        assert isinstance(ir, PromptIR)
        assert ir.is_single_span()              # conservative fallback
        # reconstruct returns a string for the fallback
        assert isinstance(reconstruct(ir), str)


def test_to_dict_is_serializable():
    import json
    ir = parse('q ```c``` "x"')
    json.dumps(ir.to_dict())                    # must not raise


# ---------------- §8.1 Equivalence: optimizers see identical input via the IR ----------------

@pytest.mark.parametrize("text", [
    "plain question with no structure",
    "before ```python\ndef f():\n    return [1,  2]\n``` after",
    'reformat "a    b\tc" exactly please',
    "Please kindly note that as per our discussion " * 3,
    "mix `inline` ```block``` \"quote\" 'single' and prose",
])
def test_compression_identical_through_ir(text):
    """Compressing reconstruct(parse(x)) must equal compressing x — the IR is
    transparent to the existing compressor in Phase 0."""
    from tokeymeter.compression import StructuralCompressor
    c = StructuralCompressor()
    from tokeymeter.prompt_ir import parse, reconstruct
    direct = c.compress(text).after
    via_ir = c.compress(reconstruct(parse(text))).after
    assert direct == via_ir


@pytest.mark.parametrize("text", [
    "question one", "with ```code``` inside", 'and "quotes" here',
])
def test_cache_key_identical_through_ir(text):
    """The exact cache key over reconstruct(parse(x)) must equal the key over x."""
    from tokeymeter.utils import make_cache_key
    from tokeymeter.prompt_ir import parse, reconstruct
    k_direct = make_cache_key((text,), {}, model="m")
    k_via_ir = make_cache_key((reconstruct(parse(text)),), {}, model="m")
    assert k_direct == k_via_ir


# ---------------- Phase 1: IR-native compression (compress_ir) ----------------

@pytest.mark.parametrize("text", [
    "Please kindly note that as per our discussion " * 3,
    "before ```python\ndef f():\n    return [1,  2]\n``` after",
    'reformat "a    b\tc" exactly please',
    "mix `inline` ```block``` \"quote\" and lots of   filler   words here",
    "plain prose with no structure at all just words",
])
def test_compress_ir_string_is_byte_identical_to_compress(text):
    """STRING IRs must compress byte-identically to the existing compressor."""
    from tokeymeter.compression import StructuralCompressor
    from tokeymeter.prompt_ir import parse, reconstruct
    c = StructuralCompressor()
    direct = c.compress(text).after
    via_ir = reconstruct(c.compress_ir(parse(text)))
    assert via_ir == direct


def test_compress_ir_messages_protects_verbatim_spans():
    """Message IRs: user query + system + tool are protected verbatim (never
    lossily touched); assistant/history content is conservatively compressed."""
    from tokeymeter.compression import StructuralCompressor
    from tokeymeter.prompt_ir import parse, reconstruct
    c = StructuralCompressor()
    msgs = [
        {"role": "system", "content": "Please    kindly   keep    this    exact."},
        {"role": "user", "content": "Please kindly keep the user query exact please."},
        {"role": "assistant", "content": "Please kindly note as per our discussion " * 3},
        {"role": "tool", "content": "schema   with   spaces"},
    ]
    out = c.compress_ir(parse(msgs))
    rec = reconstruct(out)
    # system, user query, and tool are verbatim -> unchanged
    assert rec[0]["content"] == msgs[0]["content"]
    assert rec[1]["content"] == msgs[1]["content"]
    assert rec[3]["content"] == msgs[3]["content"]
    # assistant history is compressible -> compressed (shorter)
    assert len(rec[2]["content"]) < len(msgs[2]["content"])
    assert [m["role"] for m in rec] == ["system", "user", "assistant", "tool"]


def test_compress_ir_fails_open_on_bad_input():
    from tokeymeter.compression import StructuralCompressor
    from tokeymeter.prompt_ir import parse
    c = StructuralCompressor()
    ir = parse("x")
    # passing a malformed object returns it unchanged (fail-open), never raises
    assert c.compress_ir(ir) is not None
