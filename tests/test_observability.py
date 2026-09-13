"""Regression tests for observability hardening (#2): every self-heal / fallback
path is now visible, and the degraded bus reports per-source counts so operators
can see not just THAT something degraded but exactly HOW and HOW OFTEN.

Design split:
  - anomalies / self-heals / fallbacks  -> degraded events (rare, each matters):
      compression_fallback, redis_unhealthy, redis_write, semantic_vec_disabled,
      semantic_vec_load_failed.
  - high-frequency operational signals   -> counters (aggregate, not per-event):
      MemoryStore eviction counter.
"""
import os
import random
import tempfile

import pytest

import tokeymeter
import tokeymeter.decorator as dec
from tokeymeter import degraded
from tokeymeter.storage import MemoryStore
from tokeymeter.salience import SalienceCompressor


def test_degraded_counts_are_per_source():
    degraded.clear_subscribers()
    degraded.emit_degraded("alpha", RuntimeError("x"))
    degraded.emit_degraded("alpha", RuntimeError("y"))
    degraded.emit_degraded("beta", RuntimeError("z"))
    counts = degraded.degraded_counts()
    assert counts["alpha"] == 2
    assert counts["beta"] == 1
    assert degraded.degraded_event_count() == 3
    degraded.clear_subscribers()
    assert degraded.degraded_counts() == {}        # reset clears the breakdown
    assert degraded.degraded_event_count() == 0


def test_compression_fallback_emits_degraded_event():
    degraded.clear_subscribers()
    random.seed(1)
    facts = [f"In region {i}, Q3 revenue was ${random.randint(1, 99)}M." for i in range(50)]
    doc = " ".join(["This is background context for the analysis."] * 200 + facts)
    # raw salience over-compresses (~0.83) beyond the 0.80 cap -> withheld
    _, kwargs, _ = dec._apply_compressor((), {"prompt": doc}, "prompt", None,
                                         SalienceCompressor(target_ratio=0.5))
    assert kwargs["prompt"] == doc  # original withheld
    assert degraded.degraded_counts().get("compression_fallback", 0) >= 1
    degraded.clear_subscribers()


def test_redis_unhealthy_fires_once_per_outage_edge():
    from tokeymeter.backends.redis_store import RedisStore
    from tokeymeter.backends.cipher import FernetCipher
    from cryptography.fernet import Fernet

    class Boom:
        def set(self, *a, **k): raise ConnectionError("down")
        def get(self, *a, **k): raise ConnectionError("down")
        def scan(self, *a, **k): return (0, [])

    degraded.clear_subscribers()
    rs = RedisStore(client=Boom(), namespace="t",
                    cipher=FernetCipher(Fernet.generate_key()))
    rs.set("k", {"v": 1})   # write fail: redis_write + redis_unhealthy (transition)
    rs.get("k")             # during cooldown: must NOT re-emit redis_unhealthy
    rs.get("k")
    counts = degraded.degraded_counts()
    assert counts.get("redis_write", 0) >= 1
    assert counts.get("redis_unhealthy", 0) == 1, "unhealthy must fire once per outage edge, not per call"
    degraded.clear_subscribers()


def test_memory_store_eviction_counter():
    store = MemoryStore(max_entries=100)
    for i in range(500):
        store.set(f"k{i}", i)
    st = store.stats()
    assert st["size"] == 100
    assert st["max_entries"] == 100
    assert st["evictions"] == 400   # 500 inserts - 100 retained


def test_semantic_vec_disabled_emits_on_runtime_failure():
    from tokeymeter.semantic import SemanticCache, is_vec_index_available
    if not is_vec_index_available():
        pytest.skip("sqlite-vec not available")
    import numpy as np

    degraded.clear_subscribers()
    d = tempfile.mkdtemp()
    # cache schema expects dim=3, but the encoder yields dim=4 -> the vec0 INSERT
    # fails at runtime and the backend self-heals to linear scan, emitting an event.
    c = SemanticCache(path=os.path.join(d, "sem.db"),
                      encoder=lambda t: np.array([1.0, 2.0, 3.0, 4.0], dtype="float32"),
                      dim=3)
    assert c._use_vec is True
    c.store("a prompt that triggers a vec insert", "response")
    assert c._use_vec is False, "vec mode should self-disable on insert failure"
    assert degraded.degraded_counts().get("semantic_vec_disabled", 0) >= 1
    degraded.clear_subscribers()
