"""Quality-preservation: high-stakes / do-not-optimize mode.

Guarantee: every optimization is bypassed (no cache serve/store, no semantic,
no compression); the wrapped fn always runs fresh with the prompt unaltered;
the result is never cached; the call is still recorded to audit.
"""
import pytest
import tokeymeter
from tokeymeter.storage import MemoryStore


@pytest.fixture(autouse=True)
def _mem():
    tokeymeter.set_default_store(MemoryStore())
    tokeymeter.set_default_semantic_cache(None)
    tokeymeter.set_default_redactor(None)


def test_high_stakes_decorator_never_serves_cache():
    calls = [0]
    @tokeymeter.cache(model="m", high_stakes=True)
    def hs(p):
        calls[0] += 1
        return f"r:{p}"
    hs("k"); hs("k"); hs("k")
    assert calls[0] == 3, "high_stakes must re-run every time, never serve cache"


def test_high_stakes_result_not_written_to_cache():
    @tokeymeter.cache(model="m", high_stakes=True)
    def hs(p): return "HS"
    @tokeymeter.cache(model="m")
    def normal(p): return "NORMAL"
    hs("shared")
    assert normal("shared") == "NORMAL", "high_stakes result must not populate cache"


def test_no_optimize_context_forces_bypass():
    calls = [0]
    @tokeymeter.cache(model="m")
    def ask(p):
        calls[0] += 1
        return "x"
    ask("k"); ask("k")
    assert calls[0] == 1            # cached
    with tokeymeter.no_optimize():
        ask("k"); ask("k")
    assert calls[0] == 3, "no_optimize() must bypass cache for calls in the block"
    ask("k")                         # back to normal caching outside the block
    assert calls[0] == 3


def test_high_stakes_prompt_untouched_by_compressor():
    from tokeymeter.compression import StructuralCompressor
    seen = {}
    @tokeymeter.cache(model="m", high_stakes=True, compressor=StructuralCompressor())
    def hs(prompt):
        seen["p"] = prompt
        return "ok"
    messy = "Please    note:   ```code   spaced```"
    hs(messy)
    assert seen["p"] == messy, "compression must not alter the prompt under high_stakes"


def test_high_stakes_still_audited(tmp_path):
    from tokeymeter.audit import AuditLog
    audit = AuditLog(path=str(tmp_path/"a.db"),
                     install_secret_path=str(tmp_path/"s"),
                     signing_key_path=str(tmp_path/"k"))
    audit.attach()
    try:
        @tokeymeter.cache(model="m", high_stakes=True)
        def hs(p): return "ok"
        hs("audit me")
        audit.flush(timeout=2.0)
        assert len(audit.get_entries()) >= 1, "high_stakes call must be audited"
    finally:
        audit.detach() if hasattr(audit, "detach") else None


@pytest.mark.asyncio
async def test_high_stakes_async():
    calls = [0]
    @tokeymeter.cache(model="m", high_stakes=True)
    async def hs(p):
        calls[0] += 1
        return "x"
    await hs("k"); await hs("k")
    assert calls[0] == 2, "async high_stakes must re-run every time"


@pytest.mark.asyncio
async def test_no_optimize_bypasses_cache_stream():
    calls = [0]

    @tokeymeter.cache_stream(model="m")
    async def stream(prompt):
        calls[0] += 1
        yield f"{calls[0]}:{prompt}"

    async def collect():
        return [c async for c in stream("k")]

    assert await collect() == ["1:k"]
    assert await collect() == ["1:k"]
    with tokeymeter.no_optimize():
        assert await collect() == ["2:k"]
        assert await collect() == ["3:k"]
    assert calls[0] == 3


# --- Privacy: high-stakes must never leak plaintext in event previews ---

def test_high_stakes_event_preview_is_not_plaintext():
    from tokeymeter import events as ev
    from tokeymeter import decorator as _dec
    _dec._PREVIEW_POLICY = "full"            # even with the most permissive policy
    captured = []
    unsub = ev.subscribe(lambda e: captured.append(e))
    try:
        secret = "SSN 123-45-6789 terminal diagnosis CONFIDENTIAL"
        @tokeymeter.cache(model="m", high_stakes=True)
        def critical(p): return "ok"
        critical(secret)
        e = captured[-1]
        assert secret[:15] not in (e.prompt_preview or ""), "high_stakes leaked plaintext!"
        assert (e.prompt_preview or "").startswith("sha256:"), "expected hashed preview"
    finally:
        pass


def test_no_optimize_block_preview_not_plaintext():
    from tokeymeter import events as ev
    from tokeymeter import decorator as _dec
    _dec._PREVIEW_POLICY = "full"
    captured = []
    ev.subscribe(lambda e: captured.append(e))
    secret = "highly sensitive content here 999"
    @tokeymeter.cache(model="m")
    def ask(p): return "ok"
    with tokeymeter.no_optimize():
        ask(secret)
    assert secret[:15] not in (captured[-1].prompt_preview or "")


def test_preview_policy_omit_and_hashed():
    from tokeymeter import events as ev
    from tokeymeter import decorator as _dec
    captured = []
    ev.subscribe(lambda e: captured.append(e))
    @tokeymeter.cache(model="m")
    def ask(p): return "ok"
    try:
        tokeymeter.set_event_preview_policy("omit")
        captured.clear(); ask("anything")
        assert captured[-1].prompt_preview is None
        tokeymeter.set_event_preview_policy("hashed")
        captured.clear(); ask("anything else")
        assert (captured[-1].prompt_preview or "").startswith("sha256:")
        with pytest.raises(ValueError):
            tokeymeter.set_event_preview_policy("bogus")
    finally:
        tokeymeter.set_event_preview_policy("full")   # restore default
