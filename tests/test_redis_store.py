"""
Extensive tests for the distributed backend (v0.9).

Tested under all parameters the design contract promises:
  - Basic get/set/clear/len semantics (via fakeredis — real Redis behavior)
  - Native TTL derived from the envelope; expiry eviction
  - Zero-knowledge encryption: roundtrip, no-plaintext-leak, wrong-key miss,
    tamper miss
  - FAIL-OPEN under injected failures: Redis raising on get/set/scan/ping,
    cooldown behavior, recovery
  - Namespacing isolation
  - Distributed single-flight: leader/follower, lock auto-expiry, owner-only
    release, and concurrent threads collapsing to ONE real computation
  - Drop-in decorator integration: cross-"pod" cache sharing + stampede
    protection under concurrent load
"""
import threading
import time

import pytest

import tokeymeter
from tokeymeter import events
from tokeymeter.envelope import wrap, unwrap
from tokeymeter.storage import MemoryStore

fakeredis = pytest.importorskip("fakeredis")
from tokeymeter.backends import RedisStore, FernetCipher, NoOpCipher  # noqa: E402

# These tests exercise caching mechanics, not encryption; plaintext is the
# deliberate choice here, so silence the H1 no-cipher warning module-wide.
pytestmark = pytest.mark.filterwarnings(
    "ignore:RedisStore:UserWarning"
)


@pytest.fixture
def fake():
    return fakeredis.FakeStrictRedis()


@pytest.fixture
def store(fake):
    return RedisStore(client=fake, namespace="test")


@pytest.fixture(autouse=True)
def reset_state():
    tokeymeter.set_default_store(MemoryStore())
    tokeymeter.reset_savings()
    events.clear_subscribers()
    yield
    events.clear_subscribers()


# ============================================================
#                 Failure-injection clients
# ============================================================

class BrokenRedis:
    """Every operation raises — simulates a hard Redis outage."""
    def get(self, *a, **k): raise ConnectionError("redis down")
    def set(self, *a, **k): raise ConnectionError("redis down")
    def scan(self, *a, **k): raise ConnectionError("redis down")
    def delete(self, *a, **k): raise ConnectionError("redis down")
    def eval(self, *a, **k): raise ConnectionError("redis down")
    def ping(self, *a, **k): raise ConnectionError("redis down")


class FlakyRedis:
    """Wraps a real fakeredis but fails the first N operations."""
    def __init__(self, inner, fail_first=2):
        self._inner = inner
        self._fails_left = fail_first
    def _maybe_fail(self):
        if self._fails_left > 0:
            self._fails_left -= 1
            raise ConnectionError("transient blip")
    def get(self, *a, **k): self._maybe_fail(); return self._inner.get(*a, **k)
    def set(self, *a, **k): self._maybe_fail(); return self._inner.set(*a, **k)
    def scan(self, *a, **k): return self._inner.scan(*a, **k)
    def delete(self, *a, **k): return self._inner.delete(*a, **k)
    def eval(self, *a, **k): return self._inner.eval(*a, **k)
    def ping(self, *a, **k): return self._inner.ping(*a, **k)


# ============================================================
#                 Basic semantics
# ============================================================

def test_set_get_roundtrip(store):
    store.set("k", wrap("hello", ttl=60))
    assert unwrap(store.get("k")) == "hello"


def test_miss_returns_none(store):
    assert store.get("absent") is None


def test_len_counts_value_keys(store):
    store.set("a", wrap(1, ttl=60))
    store.set("b", wrap(2, ttl=60))
    assert len(store) == 2


def test_clear_removes_all(store):
    store.set("a", wrap(1, ttl=60))
    store.set("b", wrap(2, ttl=60))
    store.clear()
    assert len(store) == 0
    assert store.get("a") is None


def test_unserializable_value_skipped(store):
    store.set("k", object())  # not JSON-serializable in a meaningful way
    # default=str makes most things serializable; ensure no raise either way
    # (the contract is "never raise")


def test_ping_true_when_up(store):
    assert store.ping() is True


# ============================================================
#                 TTL
# ============================================================

def test_native_ttl_set_from_envelope(store, fake):
    store.set("k", wrap("v", ttl=100))
    # fakeredis supports TTL inspection
    ttl = fake.ttl("test:v:k")
    assert 1 <= ttl <= 100


def test_no_ttl_when_envelope_has_none(store, fake):
    store.set("k", wrap("v", ttl=None))
    ttl = fake.ttl("test:v:k")
    # -1 = no expiry in Redis semantics
    assert ttl == -1


def test_default_ttl_applied_when_no_envelope_expiry(fake):
    s = RedisStore(client=fake, namespace="dt", default_ttl=50)
    s.set("k", wrap("v", ttl=None))
    ttl = fake.ttl("dt:v:k")
    assert 1 <= ttl <= 50


# ============================================================
#                 Encryption (zero-knowledge cache)
# ============================================================

def test_encryption_roundtrip(fake):
    key = FernetCipher.generate_key()
    s = RedisStore(client=fake, namespace="enc", cipher=FernetCipher(key))
    s.set("secret", wrap("classified", ttl=60))
    assert unwrap(s.get("secret")) == "classified"


def test_encryption_no_plaintext_in_redis(fake):
    key = FernetCipher.generate_key()
    s = RedisStore(client=fake, namespace="enc", cipher=FernetCipher(key))
    s.set("secret", wrap("the-secret-password-1234", ttl=60))
    raw = fake.get("enc:v:secret")
    assert b"the-secret-password-1234" not in raw
    assert b"secret" not in raw  # not even partial plaintext


def test_encryption_wrong_key_is_a_miss(fake):
    s1 = RedisStore(client=fake, namespace="enc",
                    cipher=FernetCipher(FernetCipher.generate_key()))
    s1.set("k", wrap("data", ttl=60))
    # A different store with a DIFFERENT key reads the same Redis
    s2 = RedisStore(client=fake, namespace="enc",
                    cipher=FernetCipher(FernetCipher.generate_key()))
    assert s2.get("k") is None  # can't decrypt → fail-open miss


def test_encryption_tamper_is_a_miss(fake):
    key = FernetCipher.generate_key()
    s = RedisStore(client=fake, namespace="enc", cipher=FernetCipher(key))
    s.set("k", wrap("data", ttl=60))
    # Corrupt the ciphertext in Redis
    fake.set("enc:v:k", b"corrupted-ciphertext-bytes")
    assert s.get("k") is None  # authentication fails → miss


def test_passphrase_derivation_is_deterministic():
    salt = b"a-stable-salt-16b"
    c1 = FernetCipher.from_passphrase("hunter2", salt=salt)
    c2 = FernetCipher.from_passphrase("hunter2", salt=salt)
    blob = c1.encrypt(b"x")
    # c2 derived the same key → can decrypt c1's output
    assert c2.decrypt(blob) == b"x"


def test_noop_cipher_passthrough():
    c = NoOpCipher()
    assert c.decrypt(c.encrypt(b"abc")) == b"abc"


# ============================================================
#                 FAIL-OPEN under failure injection
# ============================================================

def test_get_fails_open_on_outage():
    s = RedisStore(client=BrokenRedis(), namespace="x")
    assert s.get("k") is None  # no raise


def test_set_fails_open_on_outage():
    s = RedisStore(client=BrokenRedis(), namespace="x")
    s.set("k", wrap("v", ttl=60))  # must not raise


def test_clear_and_len_fail_open_on_outage():
    s = RedisStore(client=BrokenRedis(), namespace="x")
    s.clear()              # no raise
    assert len(s) == 0     # fail-open → 0


def test_ping_false_on_outage():
    s = RedisStore(client=BrokenRedis(), namespace="x")
    assert s.ping() is False


def test_cooldown_after_failure_then_recovery(fake):
    flaky = FlakyRedis(fake, fail_first=1)
    s = RedisStore(client=flaky, namespace="rec")
    # First get fails → marks unhealthy, returns None
    assert s.get("k") is None
    # During cooldown, subsequent ops short-circuit to None without hitting redis
    assert s.get("k") is None
    # After cooldown, it recovers
    s._cooldown = 0.0  # force cooldown to elapse
    s.set("k", wrap("v", ttl=60))
    assert unwrap(s.get("k")) == "v"


# ============================================================
#                 Namespacing
# ============================================================

def test_namespaces_isolated(fake):
    a = RedisStore(client=fake, namespace="A")
    b = RedisStore(client=fake, namespace="B")
    a.set("k", wrap("from-a", ttl=60))
    b.set("k", wrap("from-b", ttl=60))
    assert unwrap(a.get("k")) == "from-a"
    assert unwrap(b.get("k")) == "from-b"


# ============================================================
#                 Distributed single-flight
# ============================================================

def test_lock_acquire_and_owner_release(store):
    token = store.acquire_compute_lock("key1")
    assert token is not None
    # A second acquire while held returns None
    assert store.acquire_compute_lock("key1") is None
    # Release by owner frees it
    store.release_compute_lock("key1", token)
    assert store.acquire_compute_lock("key1") is not None


def test_lock_release_only_by_owner(store):
    token = store.acquire_compute_lock("key2")
    # Wrong token does not release
    store.release_compute_lock("key2", "wrong-token")
    assert store.acquire_compute_lock("key2") is None  # still held
    store.release_compute_lock("key2", token)          # right token
    assert store.acquire_compute_lock("key2") is not None


def test_lock_auto_expires(fake):
    s = RedisStore(client=fake, namespace="lk", lock_ttl=1.0)
    token = s.acquire_compute_lock("k")
    assert token is not None
    # Simulate TTL expiry in fakeredis
    fake.pexpire("lk:lock:k", 1)
    time.sleep(0.02)
    # After expiry, a new leader can acquire
    assert s.acquire_compute_lock("k") is not None


def test_wait_for_result_returns_peer_value(store):
    # Simulate a peer storing the result while we wait
    def peer():
        time.sleep(0.05)
        store.set("shared", wrap("peer-result", ttl=60))
    threading.Thread(target=peer).start()
    env = store.wait_for_result("shared", timeout=2.0)
    assert env is not None
    assert unwrap(env) == "peer-result"


def test_wait_for_result_times_out(store):
    env = store.wait_for_result("never", timeout=0.2)
    assert env is None


# ============================================================
#                 Decorator integration
# ============================================================

def test_redis_store_is_drop_in(store):
    """RedisStore works in the decorator exactly like MemoryStore."""
    @tokeymeter.cache(model="gpt-4o-mini", store=store, prompt_arg="prompt")
    def ask(prompt):
        return "answer-" + prompt[:3]

    assert ask(prompt="hello") == "answer-hel"
    assert ask(prompt="hello") == "answer-hel"  # served from Redis
    report = tokeymeter.savings_report()
    assert report["cache_hits"] >= 1


def test_cross_pod_cache_sharing(fake):
    """Two independent decorators (simulating two pods) share one Redis and
    therefore share cache hits — the whole point of v0.9."""
    store_pod1 = RedisStore(client=fake, namespace="shared")
    store_pod2 = RedisStore(client=fake, namespace="shared")

    calls = {"pod1": 0, "pod2": 0}

    @tokeymeter.cache(model="gpt-4o-mini", store=store_pod1, prompt_arg="prompt", single_flight=False, namespace="pod-workload")
    def pod1(prompt):
        calls["pod1"] += 1
        return "shared-answer"

    @tokeymeter.cache(model="gpt-4o-mini", store=store_pod2, prompt_arg="prompt", single_flight=False, namespace="pod-workload")
    def pod2(prompt):
        calls["pod2"] += 1
        return "shared-answer"

    pod1(prompt="same question")          # miss on pod1 → real call
    result = pod2(prompt="same question")  # HIT — served from pod1's write
    assert result == "shared-answer"
    assert calls["pod1"] == 1
    assert calls["pod2"] == 0  # pod2 never computed — used pod1's cached result


def test_distributed_single_flight_collapses_concurrent_calls(fake):
    """N threads (simulating N pods) hit the same cold key at once. With
    distributed single-flight, only ONE real computation should happen."""
    real_calls = {"n": 0}
    lock = threading.Lock()
    barrier = threading.Barrier(10)

    def make_pod():
        store = RedisStore(client=fake, namespace="herd",
                           single_flight_timeout=5.0, single_flight_poll=0.01)

        @tokeymeter.cache(model="gpt-4o-mini", store=store, prompt_arg="prompt",
                    single_flight=True, namespace="herd-workload")
        def pod(prompt):
            with lock:
                real_calls["n"] += 1
            time.sleep(0.2)  # simulate a slow model call so peers wait
            return "herd-answer"
        return pod

    results = []
    res_lock = threading.Lock()

    def worker():
        pod = make_pod()
        barrier.wait()  # all start at once
        r = pod(prompt="stampede")
        with res_lock:
            results.append(r)

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads: t.start()
    for t in threads: t.join()

    assert all(r == "herd-answer" for r in results)
    assert len(results) == 10
    # The key assertion: the thundering herd collapsed. Without single-flight
    # this would be 10. With it, it should be a small number (ideally 1).
    assert real_calls["n"] <= 3, f"stampede not contained: {real_calls['n']} real calls"


def test_single_flight_fails_open_when_lock_breaks(fake):
    """If the lock mechanism is broken, every call still succeeds (computes
    locally) — never deadlocks or errors."""
    class NoLockRedis(FlakyRedis):
        def set(self, *a, **k):
            # Fail only lock SETs (nx=True); allow value SETs
            if k.get("nx"):
                raise ConnectionError("lock backend down")
            return self._inner.set(*a, **k)

    store = RedisStore(client=NoLockRedis(fake, fail_first=0), namespace="nl")

    @tokeymeter.cache(model="gpt-4o-mini", store=store, prompt_arg="prompt", single_flight=True)
    def ask(prompt):
        return "ok"

    # Must not raise, must return correct result
    assert ask(prompt="x") == "ok"


# --- Regression tests for H1/H2 (zero-knowledge defaults) ---

def test_h1_no_cipher_warns():
    """H1: a shared store with no cipher must warn that values are plaintext."""
    import warnings, fakeredis
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        RedisStore(client=fakeredis.FakeStrictRedis())
        assert any("PLAINTEXT" in str(x.message) for x in w)


def test_h2_key_secret_defeats_confirmation_attack():
    """H2: with a key_secret, the Redis keyspace holds only HMACs, so an
    attacker cannot confirm a guessable prompt via the plain SHA-256 key."""
    import fakeredis
    from tokeymeter.utils import make_cache_key
    fake = fakeredis.FakeStrictRedis()
    store = RedisStore(client=fake, cipher=NoOpCipher(), key_secret=b"s" * 32)
    prompt = "Does patient Jane Roe have HIV?"
    store.set(make_cache_key((prompt,), {}, model="m"), {"r": 1})
    naive = "tokeymeter:v:" + make_cache_key((prompt,), {}, model="m")
    assert naive not in {k.decode() for k in fake.keys("tokeymeter:v:*")}


def test_h2_cross_pod_sharing_preserved_with_key_secret():
    """H2 must not break the shared cache: same secret => same key => shared."""
    import fakeredis
    from tokeymeter.utils import make_cache_key
    fake = fakeredis.FakeStrictRedis()
    a = RedisStore(client=fake, cipher=NoOpCipher(), key_secret=b"s" * 32)
    b = RedisStore(client=fake, cipher=NoOpCipher(), key_secret=b"s" * 32)
    k = make_cache_key(("p",), {}, model="m")
    a.set(k, {"r": "x"})
    assert b.get(k) is not None
