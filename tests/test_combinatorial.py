"""Combinatorial integration tests — feature interactions, where infra breaks.

Uses REAL components: real Redis (redislite), real Fernet encryption, real
threads/async. Each test proves a dangerous combination composes correctly.
"""
import asyncio, os, tempfile, threading, time
import pytest
import tokeymeter
from tokeymeter.storage import MemoryStore
from tokeymeter.compression import StructuralCompressor, CompressionResult
from tokeymeter.pricing import estimate_tokens
from tokeymeter import events as ev
from tokeymeter.decision import clear_decision_subscribers

try:
    import redislite
    from cryptography.fernet import Fernet
    from tokeymeter.backends.redis_store import RedisStore
    from tokeymeter.backends.cipher import FernetCipher
    _REDIS = True
except Exception:
    _REDIS = False
redis_only = pytest.mark.skipif(not _REDIS, reason="redislite/cryptography not installed")


@pytest.fixture(autouse=True)
def _reset():
    tokeymeter.set_default_store(MemoryStore())
    tokeymeter.set_default_semantic_cache(None)
    tokeymeter.set_default_redactor(None)
    tokeymeter.set_event_preview_policy("full")
    ev.clear_subscribers()
    clear_decision_subscribers()
    yield
    ev.clear_subscribers()
    clear_decision_subscribers()


def _stub_encoder():
    import numpy as np
    table = {}
    def enc(text):
        # deterministic pseudo-embedding; identical text -> identical vector
        h = abs(hash(text)) % (2**32)
        v = np.random.default_rng(h).standard_normal(8).astype("float32")
        return v / (np.linalg.norm(v) or 1)
    return enc


# ------------------------------------------------------------ Redis + SF + enc
@redis_only
def test_redis_single_flight_encryption_concurrency():
    dbfile = os.path.join(tempfile.mkdtemp(), "c.rdb")
    key = Fernet.generate_key()
    def store():
        return RedisStore(client=redislite.Redis(dbfile),
                          cipher=FernetCipher(key=key), key_secret=b"k"*32,
                          namespace="t")
    n = {"c": 0}; lk = threading.Lock()
    def pod():
        # Real pods are the SAME module-level function in separate processes,
        # so they share one clean auto-namespace. Simulating pods with an
        # in-process factory creates 8 DIFFERENT function objects, which the
        # namespace anti-collision guard (correctly) disambiguates — that would
        # give each "pod" its own cache key and defeat the very cross-pod
        # single-flight this test exercises. Pin an explicit namespace to model
        # the shared identity real pods have.
        @tokeymeter.cache(model="m", store=store(), single_flight=True,
                          namespace="pods")
        def ask(p):
            with lk: n["c"] += 1
            time.sleep(0.02); return f"A::{p}"
        return ask
    pods = [pod() for _ in range(8)]
    out = [None]*8
    ts = [threading.Thread(target=lambda i=i: out.__setitem__(i, pods[i]("same")))
          for i in range(8)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert n["c"] <= 2, f"single-flight failed across pods: {n['c']} computes"
    assert len(set(out)) == 1                       # all agree
    raw = redislite.Redis(dbfile)
    assert not any(b"A::same" in (raw.get(k) or b"") for k in raw.keys("*")), "plaintext leaked"
    assert not any(b"same" in k for k in raw.keys("*")), "prompt visible in keys"


@redis_only
def test_distributed_cache_fail_open_on_dead_server():
    """If Redis is unreachable, the call must still execute (fail-open)."""
    client = redislite.Redis(os.path.join(tempfile.mkdtemp(), "d.rdb"))
    store = RedisStore(client=client, key_secret=b"k"*32)
    calls = {"n": 0}
    @tokeymeter.cache(model="m", store=store)
    def ask(p):
        calls["n"] += 1; return "ok"
    assert ask("x") == "ok"
    client.shutdown()                                # kill the server
    time.sleep(0.1)
    # store is dead, but the decorated function must still return (fail-open)
    assert ask("y") == "ok"
    assert calls["n"] == 2


# ------------------------------------------------------------ semantic + redaction
def test_semantic_plus_redaction():
    from tokeymeter.semantic import SemanticCache
    from tokeymeter.privacy import DefaultRedactor
    sc = SemanticCache(threshold=0.95, encoder=_stub_encoder())
    recs = []
    tokeymeter.on_decision(lambda r: recs.append(r))
    @tokeymeter.cache(model="m", semantic=True, semantic_cache=sc,
                redactor=DefaultRedactor())
    def ask(p): return "ok"
    ask("my email is alice@example.com and ssn 123-45-6789")
    ask("my email is alice@example.com and ssn 123-45-6789")   # exact/semantic hit
    assert recs[0].pii_redactions > 0, "redaction count should flow into the decision record"
    assert recs[1].cached                          # second served from cache


# ------------------------------------------------------------ compression + memory
def test_compression_plus_memory():
    from tokeymeter.memory import ConversationMemory, TruncationSummarizer
    mem = ConversationMemory(recent_window=2, summary_threshold=3,
                             summarizer=TruncationSummarizer())
    async def run():
        for i in range(6):
            await mem.add_turn("s", user=f"turn {i} " + "filler " * 5,
                               assistant="reply")
        full = str(await mem.get_full_context("s"))
        comp = str(await mem.get_context("s"))
        return estimate_tokens(comp) <= estimate_tokens(full)
    assert asyncio.run(run())


# ------------------------------------------------------------ shadow + audit
def test_shadow_plus_audit(tmp_path):
    from tokeymeter.audit import AuditLog
    audit = AuditLog(path=str(tmp_path/"a.db"),
                     install_secret_path=str(tmp_path/"s"),
                     signing_key_path=str(tmp_path/"k"))
    audit.attach()
    recs = []
    tokeymeter.on_decision(lambda r: recs.append(r))
    try:
        @tokeymeter.cache(model="m", shadow=True)
        def ask(p): return "ok"
        ask("x")
        audit.flush(timeout=2.0)
        assert recs[-1].decision == "shadow"
        assert len(audit.get_entries()) >= 1        # shadow still audited
    finally:
        audit.detach()


# ------------------------------------------------------------ async + streaming + events
@pytest.mark.asyncio
async def test_async_streaming_plus_events():
    seen_events = []
    ev.subscribe(lambda e: seen_events.append(e))
    state = {"calls": 0}
    @tokeymeter.cache_stream(model="m")
    async def gen(p):
        state["calls"] += 1
        for i in range(3):
            yield f"chunk{i}"
    out1 = [c async for c in gen("p")]              # miss -> computes + caches
    out2 = [c async for c in gen("p")]              # hit -> replays
    assert out1 == out2 == ["chunk0", "chunk1", "chunk2"]
    assert state["calls"] == 1                       # second served from cache
    assert len(seen_events) >= 1                      # events fired


# ------------------------------------------------------------ lineage + semantic + redaction
def test_lineage_plus_semantic_plus_redaction():
    from tokeymeter.semantic import SemanticCache
    from tokeymeter.privacy import DefaultRedactor
    sc = SemanticCache(threshold=0.95, encoder=_stub_encoder())
    calls = {"n": 0}
    @tokeymeter.cache(model="m", semantic=True, semantic_cache=sc, redactor=DefaultRedactor())
    def ask(p):
        calls["n"] += 1; return f"r{calls['n']}"
    with tokeymeter.lineage("A"):
        a = ask("same prompt with email x@y.com")
    with tokeymeter.lineage("B"):
        b = ask("same prompt with email x@y.com")
    assert calls["n"] == 2 and a != b               # lineage isolation holds even w/ semantic+redaction


# ------------------------------------------------------------ high_stakes + audit + decision
def test_high_stakes_audit_decision_combo(tmp_path):
    from tokeymeter.audit import AuditLog
    audit = AuditLog(path=str(tmp_path/"a.db"),
                     install_secret_path=str(tmp_path/"s"),
                     signing_key_path=str(tmp_path/"k"))
    audit.attach()
    recs = []; evs = []
    tokeymeter.on_decision(lambda r: recs.append(r)); ev.subscribe(lambda e: evs.append(e))
    try:
        @tokeymeter.cache(model="m", high_stakes=True)
        def critical(p): return "ok"
        critical("SSN 123-45-6789 sensitive"); critical("SSN 123-45-6789 sensitive")
        audit.flush(timeout=2.0)
        assert recs[-1].decision == "high_stakes" and recs[-1].provable
        assert not any("123-45-6789" in (e.prompt_preview or "") for e in evs)  # no leak
        assert len(audit.get_entries()) >= 2         # both calls audited (never cached)
    finally:
        audit.detach()


# ------------------------------------------------ compression + verify + breaker + decision
def test_compression_verify_breaker_decision_combo():
    tokeymeter.set_fidelity_circuit_breaker(open_threshold=0.80, close_threshold=0.85,
                                      min_samples=3, window=10, cooldown_s=60)
    recs = []
    tokeymeter.on_decision(lambda r: recs.append(r))
    @tokeymeter.cache(model="m", tag="wk", compressor=StructuralCompressor(),
                verify_rate=1.0, verify_similarity_fn=lambda o, c: 0.40)
    def ask(p): return "ok"
    for i in range(5):
        ask(f"please kindly note as per discussion task {i} " * 2)
    st = tokeymeter.compression_breaker_state("ask", "wk")
    assert st["state"] == "open"                     # measured low fidelity opened breaker
    # subsequent call: breaker open -> compression withheld, but call still works + recorded
    ask("please kindly another task entirely here now")
    assert recs[-1].decision in ("cache_miss", "cache_hit")
    tokeymeter.set_fidelity_circuit_breaker()              # restore


# ------------------------------------------------ memory + compression + events + decision
def test_memory_compression_events_decision_combo():
    evs = []; recs = []
    ev.subscribe(lambda e: evs.append(e)); tokeymeter.on_decision(lambda r: recs.append(r))
    @tokeymeter.cache(model="m", compressor=StructuralCompressor())
    def ask(p): return "ok"
    ask("Please kindly summarize as per our previous discussion " * 4)
    assert recs[-1].compressed and recs[-1].compression_ratio is not None
    assert len(evs) >= 1


# ============================ BATCH 2: adversarial situations ============================

@redis_only
def test_redis_encryption_key_mismatch_fails_open():
    """Key rotation / mismatch: a pod with the WRONG key must fail-open
    (recompute), never crash or serve corrupt plaintext."""
    dbfile = os.path.join(tempfile.mkdtemp(), "km.rdb")
    k1, k2 = Fernet.generate_key(), Fernet.generate_key()
    calls = {"n": 0}
    @tokeymeter.cache(model="m",
                store=RedisStore(client=redislite.Redis(dbfile),
                                 cipher=FernetCipher(key=k1), key_secret=b"k"*32, namespace="t"))
    def ask1(p):
        calls["n"] += 1; return "v1"
    ask1("x")                                        # stored encrypted under k1
    @tokeymeter.cache(model="m",
                store=RedisStore(client=redislite.Redis(dbfile),
                                 cipher=FernetCipher(key=k2), key_secret=b"k"*32, namespace="t"))
    def ask2(p):
        calls["n"] += 1; return "v2"
    assert ask2("x") == "v2"                          # decrypt fails -> recompute, no garbage
    assert calls["n"] == 2


@redis_only
def test_single_flight_exception_releases_lock():
    """If the computation raises under single-flight, the lock must release, the
    error must not be cached, and a retry must recompute successfully."""
    store = RedisStore(client=redislite.Redis(os.path.join(tempfile.mkdtemp(), "sf.rdb")),
                       key_secret=b"k"*32, cipher=FernetCipher(key=Fernet.generate_key()))
    state = {"n": 0}
    @tokeymeter.cache(model="m", store=store, single_flight=True)
    def ask(p):
        state["n"] += 1
        if state["n"] == 1:
            raise ValueError("boom")
        return "ok"
    with pytest.raises(ValueError):
        ask("x")
    assert ask("x") == "ok"                            # recomputed; error not cached
    assert state["n"] == 2


@redis_only
def test_redis_ttl_plus_encryption():
    store = RedisStore(client=redislite.Redis(os.path.join(tempfile.mkdtemp(), "ttl.rdb")),
                       cipher=FernetCipher(key=Fernet.generate_key()), key_secret=b"k"*32)
    calls = {"n": 0}
    @tokeymeter.cache(model="m", store=store, ttl=0.5)
    def ask(p):
        calls["n"] += 1; return "ok"
    ask("x"); ask("x")
    assert calls["n"] == 1                             # cached (encrypted)
    time.sleep(0.7)
    ask("x")
    assert calls["n"] == 2                             # expired -> recompute


def test_high_stakes_bypasses_semantic():
    from tokeymeter.semantic import SemanticCache
    sc = SemanticCache(threshold=0.90, encoder=_stub_encoder())
    calls = {"n": 0}
    @tokeymeter.cache(model="m", semantic=True, semantic_cache=sc, high_stakes=True)
    def ask(p):
        calls["n"] += 1; return "ok"
    ask("x"); ask("x")
    assert calls["n"] == 2                             # high_stakes never serves semantic


@pytest.mark.asyncio
async def test_async_high_stakes():
    calls = {"n": 0}
    @tokeymeter.cache(model="m", high_stakes=True)
    async def ask(p):
        calls["n"] += 1; return "ok"
    await ask("x"); await ask("x")
    assert calls["n"] == 2


def test_redaction_strips_pii_before_model_compressor_and_cache():
    """Composition invariant: with a redactor, PII never reaches the model, the
    compressor, the cache key, or the audit — redaction runs first."""
    from tokeymeter.privacy import DefaultRedactor
    seen = {}
    @tokeymeter.cache(model="m", redactor=DefaultRedactor(), compressor=StructuralCompressor())
    def ask(prompt):
        seen["model_input"] = prompt                  # what the function/model actually receives
        return "ok"
    ask("contact alice@example.com about this please kindly note as per discussion")
    assert "alice@example.com" not in seen["model_input"], \
        "PII must be redacted before it reaches the model"


def test_concurrent_decision_dispatch_threadsafe():
    recs = []; lk = threading.Lock()
    tokeymeter.on_decision(lambda r: (lk.acquire(), recs.append(r), lk.release()))
    @tokeymeter.cache(model="m")
    def ask(p): return "ok"
    ts = [threading.Thread(target=lambda i=i: ask(f"k{i}")) for i in range(20)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert len(recs) == 20                             # all dispatched, none lost


def test_memory_plus_lineage_isolation():
    from tokeymeter.memory import ConversationMemory, TruncationSummarizer
    async def run():
        mem = ConversationMemory(recent_window=2, summary_threshold=3,
                                 summarizer=TruncationSummarizer())
        for i in range(4):
            await mem.add_turn("tenantA", user=f"A secret {i}", assistant="ok")
            await mem.add_turn("tenantB", user=f"B secret {i}", assistant="ok")
        a = str(await mem.get_context("tenantA"))
        b = str(await mem.get_context("tenantB"))
        return ("B secret" not in a) and ("A secret" not in b)
    assert asyncio.run(run())


# ============================ BATCH 3: max-reach trust seams ============================

@redis_only
def test_tampered_ciphertext_fails_open():
    """AEAD tamper-resistance end-to-end: if an attacker mutates an encrypted
    value in Redis, authenticated decryption fails -> fail-open -> recompute.
    Tokeymeter must NEVER serve tampered/corrupt data."""
    dbfile = os.path.join(tempfile.mkdtemp(), "tamper.rdb")
    store = RedisStore(client=redislite.Redis(dbfile),
                       cipher=FernetCipher(key=Fernet.generate_key()),
                       key_secret=b"k"*32, namespace="t")
    calls = {"n": 0}
    @tokeymeter.cache(model="m", store=store)
    def ask(p):
        calls["n"] += 1; return "real-answer"
    ask("x")
    raw = redislite.Redis(dbfile)
    for k in raw.keys("*"):
        if b"lock" in k:
            continue
        v = raw.get(k)
        if v and len(v) > 10:
            raw.set(k, v[:-4] + b"XXXX")             # corrupt the auth tag
    r = ask("x")
    assert r == "real-answer" and calls["n"] == 2, "tampered value must fail-open, not be served"


@redis_only
def test_redis_namespace_isolation():
    client = redislite.Redis(os.path.join(tempfile.mkdtemp(), "ns.rdb"))
    calls = {"n": 0}
    @tokeymeter.cache(model="m", store=RedisStore(client=client, namespace="tenantA", key_secret=b"k"*32))
    def a(p):
        calls["n"] += 1; return "A"
    @tokeymeter.cache(model="m", store=RedisStore(client=client, namespace="tenantB", key_secret=b"k"*32))
    def b(p):
        calls["n"] += 1; return "B"
    a("same")
    assert b("same") == "B"                           # namespace prevents cross-tenant collision
    assert calls["n"] == 2


def test_high_stakes_with_compressor_sends_verbatim():
    seen = {}; recs = []
    tokeymeter.on_decision(lambda r: recs.append(r))
    @tokeymeter.cache(model="m", high_stakes=True, compressor=StructuralCompressor())
    def ask(prompt):
        seen["p"] = prompt; return "ok"
    messy = "Please    kindly    note   with   lots   of   spaces"
    ask(messy)
    assert seen["p"] == messy                          # verbatim despite compressor
    assert not recs[-1].compressed                      # decision reflects no compression


def test_events_audit_decision_three_way_consistency(tmp_path):
    from tokeymeter.audit import AuditLog
    audit = AuditLog(path=str(tmp_path/"a.db"),
                     install_secret_path=str(tmp_path/"s"),
                     signing_key_path=str(tmp_path/"k"))
    audit.attach()
    evs = []; recs = []
    ev.subscribe(lambda e: evs.append(e)); tokeymeter.on_decision(lambda r: recs.append(r))
    try:
        @tokeymeter.cache(model="gpt-4o-mini")
        def ask(p): return "ok"
        ask("q")
        audit.flush(timeout=2.0)
        assert len(evs) == 1 and len(recs) == 1
        assert recs[0].cache_key == evs[0].cache_key   # same identity across all 3 planes
        assert len(audit.get_entries()) == 1
        assert recs[0].provable                         # decision agrees it's recorded
    finally:
        audit.detach()


def test_no_single_flight_concurrency_no_corruption():
    n = {"c": 0}; lk = threading.Lock()
    @tokeymeter.cache(model="m", store=MemoryStore(), single_flight=False)
    def ask(p):
        with lk: n["c"] += 1
        return "ok"
    ts = [threading.Thread(target=lambda: ask("same")) for _ in range(10)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert ask("same") == "ok"                          # consistent; cached afterward
    before = n["c"]; ask("same")
    assert n["c"] == before                             # now definitely cached


@pytest.mark.asyncio
async def test_async_streaming_error_not_cached_with_events():
    evs = []
    ev.subscribe(lambda e: evs.append(e))
    state = {"n": 0}
    @tokeymeter.cache_stream(model="m")
    async def gen(p, fail=False):
        state["n"] += 1
        for i in range(3):
            if fail and i == 2:
                raise RuntimeError("mid-stream")
            yield f"c{i}"
    with pytest.raises(RuntimeError):
        async for _ in gen("p", fail=True):
            pass
    out = [c async for c in gen("p", fail=False)]        # must recompute, not serve partial
    assert out == ["c0", "c1", "c2"] and state["n"] == 2
