"""Property-based fuzzing (Phase-1 item 3).

Where the stress battery throws hostile *cases*, hypothesis explores hostile
*space* — thousands of generated inputs against invariants that must hold for
ALL input, not just the ones we thought to write. These five surfaces are the
ones a stranger feeds untrusted data through the moment we ship to PyPI, so
each carries a machine-checked contract:

  • SecretScanner — a firewall that crashes on hostile input is a firewall
    that can be bypassed. Never raises; result stays within the byte cap;
    every finding span lies inside the input; NO finding carries a raw value.
  • Envelope — wrap/unwrap is self-inverse for JSON-able payloads; the value
    survives byte-for-byte.
  • Id grammar — the regex shared by identity + keys + ontology accepts
    exactly the grammar and rejects everything else, proven both directions.
  • safe_compress — never raises on any text; ratio ∈ (0, 1]; verdict fields
    always legal.
  • Cache keys — deterministic for identical inputs; structural variants
    (suffix/prefix/whitespace/model) never collide.

Profile 'tokeymeter' (conftest): max_examples=200, deadline=None.
"""
import json

import pytest

hypothesis = pytest.importorskip("hypothesis")   # dev-only dep; skip if absent
from hypothesis import given, strategies as st   # noqa: E402

from tokeymeter.content.secrets import SecretScanner, ScanResult, Finding  # noqa: E402
from tokeymeter.envelope import wrap, unwrap      # noqa: E402
from tokeymeter.identity import _ID_RE            # noqa: E402
from tokeymeter.compression import safe_compress, CompressionResult  # noqa: E402
from tokeymeter.utils import make_cache_key       # noqa: E402


# ═══ 1. SecretScanner — the security-critical surface ════════════════════
_scanner = SecretScanner()

# planted real-shaped secrets, to be embedded at arbitrary offsets so the
# "no raw value on any finding" property is tested against ACTUAL detections,
# not just benign noise that produces zero findings.
_PLANTS = [
    "AKIAIOSFODNN7EXAMPLE",
    "sk-live-AAAA1111BBBB2222CCCC3333DDDD",
    "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
]


def _assert_scan_invariants(text: str, res: ScanResult):
    assert isinstance(res, ScanResult)
    # byte-cap honored: never scans beyond the 16MB guarantee
    assert res.bytes_scanned <= 16 * 1024 * 1024
    for f in res.findings:
        assert isinstance(f, Finding)
        s, e = f.span
        # span lies strictly inside the scanned text
        assert 0 <= s <= e <= len(text), f"span {f.span} out of bounds len={len(text)}"
        # content-blind: the finding's recordable descriptor carries NO value
        desc = f.redacted_descriptor()
        assert "value" not in desc and "secret" not in desc
        assert set(desc) == {"type", "severity", "confidence", "offset", "length"}
        # placeholder is a label, never the secret substring itself
        assert text[s:e] not in f.placeholder_label


@given(st.text(max_size=4000))
def test_scanner_never_raises_on_arbitrary_text(s):
    res = _scanner.scan(s)
    _assert_scan_invariants(s, res)


@given(st.binary(max_size=4000))
def test_scanner_never_raises_on_decoded_bytes(b):
    # arbitrary bytes → text via lossy decode (the real ingestion path)
    s = b.decode("utf-8", "replace")
    res = _scanner.scan(s)
    _assert_scan_invariants(s, res)


@given(
    prefix=st.text(max_size=2000),
    plant=st.sampled_from(_PLANTS),
    suffix=st.text(max_size=2000),
)
def test_scanner_finds_planted_secret_without_leaking_it(prefix, plant, suffix):
    text = prefix + plant + suffix
    res = _scanner.scan(text)
    _assert_scan_invariants(text, res)
    # the plant sits at a known offset; SOME finding's span must cover it
    # (position-independent detection), and no descriptor may carry the value
    plant_start = len(prefix)
    covered = any(f.span[0] <= plant_start < f.span[1] for f in res.findings)
    assert covered, "planted secret not detected"
    blob = json.dumps([f.redacted_descriptor() for f in res.findings])
    assert plant not in blob, "raw secret leaked into a recordable descriptor"


@given(st.text(alphabet="\u200b\u200c\u200d\ufeff \t\n", max_size=500))
def test_scanner_survives_zero_width_and_whitespace_storms(s):
    # zero-width / control-heavy input drives the normalization index-map path
    res = _scanner.scan(s)
    _assert_scan_invariants(s, res)


# ═══ 2. Envelope — round-trip fidelity ═══════════════════════════════════
_json_leaves = (
    st.none() | st.booleans() | st.integers()
    | st.floats(allow_nan=False, allow_infinity=False)
    | st.text(max_size=200)
)
_json_values = st.recursive(
    _json_leaves,
    lambda children: st.lists(children, max_size=8)
    | st.dictionaries(st.text(max_size=40), children, max_size=8),
    max_leaves=30,
)


@given(_json_values)
def test_envelope_roundtrip_is_identity_for_json_payloads(value):
    restored = unwrap(wrap(value))
    assert restored == value


@given(_json_values, st.floats(min_value=1.0, max_value=1e6))
def test_envelope_roundtrip_with_future_ttl_preserves_value(value, ttl):
    # a not-yet-expired TTL must not alter the payload
    restored = unwrap(wrap(value, ttl=ttl))
    assert restored == value


@given(st.text(max_size=100))
def test_unwrap_passes_through_non_envelope_values(s):
    # legacy/raw values (not envelope-shaped) come back as-is, never crash
    assert unwrap(s) == s


# ═══ 3. Id grammar — accept/reject, both directions ══════════════════════
# generator for strings that ARE valid ids: first char alnum, then up to 127
# of [alnum . _ : -]
_id_first = st.sampled_from(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")
_id_rest = st.text(
    alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._:-",
    max_size=127)


@given(_id_first, _id_rest)
def test_valid_ids_are_accepted(first, rest):
    candidate = first + rest
    assert _ID_RE.match(candidate), f"rejected a valid id: {candidate!r}"


@given(st.text(max_size=200))
def test_id_grammar_matches_only_the_grammar(s):
    # the regex's own verdict is the oracle; assert it's internally consistent
    # with the grammar definition (no partial matches, length bound honored)
    m = _ID_RE.match(s)
    if m is None:
        return
    # a match must consume the ENTIRE string (anchored ^…$) and obey length
    assert m.group(0) == s
    assert 1 <= len(s) <= 128
    assert s[0].isalnum()
    assert all(c.isalnum() or c in "._:-" for c in s)


@given(st.sampled_from(["", " ", "\n", "a b", "café", "x/y", "a@b",
                        "A" * 129, "-lead", ".dot", ":colon"]))
def test_known_invalid_ids_are_rejected(bad):
    # explicit rejection corpus: empty, whitespace, non-ascii, illegal chars,
    # over-length, illegal leading char
    assert _ID_RE.match(bad) is None, f"accepted an invalid id: {bad!r}"


# ═══ 4. safe_compress — never-raise + legal-verdict ══════════════════════
class _HostileCompressor:
    """A compressor that always raises — safe_compress must absorb it."""
    def compress(self, text):
        raise RuntimeError("compressor exploded")


@given(st.text(max_size=3000))
def test_safe_compress_never_raises_and_verdict_is_legal(s):
    for comp in (None, _HostileCompressor()):
        res = safe_compress(comp, s)
        assert isinstance(res, CompressionResult)
        assert 0.0 < res.ratio <= 1.0, f"ratio out of (0,1]: {res.ratio}"
        assert isinstance(res.safe, bool)
        assert isinstance(res.method, str) and res.method
        # never-raise contract: a hostile compressor yields safe=False, and
        # the text is preserved unchanged (fail-open to the original prompt)
        if isinstance(comp, _HostileCompressor) and s:
            assert res.safe is False
            assert res.after == s


# ═══ 5. Cache keys — determinism + no structural collision ═══════════════
@given(st.text(max_size=500), st.text(max_size=40))
def test_cache_key_is_deterministic(prompt, model):
    k1 = make_cache_key((prompt,), {}, model=model or "_default")
    k2 = make_cache_key((prompt,), {}, model=model or "_default")
    assert k1 == k2 and len(k1) == 64          # sha256 hex


@given(st.text(min_size=1, max_size=400))
def test_structural_variants_do_not_collide(prompt):
    base = make_cache_key((prompt,), {}, model="m")
    variants = {
        "suffix":     make_cache_key((prompt + "x",), {}, model="m"),
        "prefix":     make_cache_key(("x" + prompt,), {}, model="m"),
        "trail_ws":   make_cache_key((prompt + " ",), {}, model="m"),
        "lead_ws":    make_cache_key((" " + prompt,), {}, model="m"),
        "other_model": make_cache_key((prompt,), {}, model="m2"),
        "kwarg_added": make_cache_key((prompt,), {"t": 1}, model="m"),
    }
    for name, v in variants.items():
        assert v != base, f"structural variant '{name}' collided with base"


@given(st.text(max_size=300), st.text(max_size=300))
def test_distinct_prompts_distinct_keys(a, b):
    ka = make_cache_key((a,), {}, model="m")
    kb = make_cache_key((b,), {}, model="m")
    assert (ka == kb) == (a == b)              # equal keys IFF equal prompts
