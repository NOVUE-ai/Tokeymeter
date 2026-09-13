"""Conservative compression — measured-fidelity circuit breaker.

The breaker reacts ONLY to measured fidelity (verify_rate similarity samples).
It never predicts quality. With verify_rate==0 it has no data and never trips.
"""
import time
import pytest
import tokeymeter
from tokeymeter.storage import MemoryStore
from tokeymeter.compression import CompressionResult
from tokeymeter.pricing import estimate_tokens
from tokeymeter._fidelity import _CompressionFidelityBreaker
from tokeymeter import decorator as _dec
from tokeymeter import _fidelity as _fid


# ---------------- Unit: state machine ----------------

def test_breaker_opens_on_low_measured_fidelity():
    b = _CompressionFidelityBreaker(open_threshold=0.80, min_samples=3, window=10)
    for _ in range(3):
        b.record("k", 0.50)
    assert b.state("k")["state"] == "open"
    assert b.should_compress("k") is False


def test_breaker_does_not_trip_before_min_samples():
    b = _CompressionFidelityBreaker(open_threshold=0.80, min_samples=5)
    for _ in range(4):
        b.record("k", 0.10)         # terrible, but below min_samples
    assert b.state("k")["state"] == "closed"
    assert b.should_compress("k") is True


def test_breaker_recovers_after_cooldown_on_fresh_good_fidelity():
    b = _CompressionFidelityBreaker(open_threshold=0.80, close_threshold=0.85,
                                    min_samples=3, cooldown_s=0.2)
    for _ in range(3):
        b.record("k", 0.50)
    assert b.state("k")["state"] == "open"
    time.sleep(0.25)
    assert b.should_compress("k") is True          # half-open probe
    for _ in range(3):
        b.record("k", 0.99)
    assert b.state("k")["state"] == "closed"       # recovered on fresh data


def test_breaker_reopens_if_probe_still_bad():
    b = _CompressionFidelityBreaker(open_threshold=0.80, min_samples=3, cooldown_s=0.2)
    for _ in range(3):
        b.record("k", 0.40)
    time.sleep(0.25); b.should_compress("k")        # -> half_open
    for _ in range(3):
        b.record("k", 0.40)
    assert b.state("k")["state"] == "open"


def test_breaker_per_workload_isolation():
    b = _CompressionFidelityBreaker(open_threshold=0.80, min_samples=3)
    for _ in range(3):
        b.record("A", 0.50)
    assert b.should_compress("A") is False
    assert b.should_compress("B") is True           # independent workload unaffected


def test_breaker_disabled_never_trips():
    b = _CompressionFidelityBreaker(enabled=False, min_samples=1)
    b.record("k", 0.0)
    assert b.should_compress("k") is True


# ---------------- Integration: through the decorator ----------------

class _Compressor:
    def compress(self, text):
        kept = text.replace("please ", "").replace("kindly ", "")
        return CompressionResult(before=text, after=kept,
            tokens_before=estimate_tokens(text), tokens_after=estimate_tokens(kept),
            ratio=estimate_tokens(kept) / max(1, estimate_tokens(text)),
            duration_ms=0.0, method="t", safe=True)


@pytest.fixture(autouse=True)
def _reset():
    tokeymeter.set_default_store(MemoryStore())
    tokeymeter.set_default_semantic_cache(None)
    tokeymeter.set_default_redactor(None)
    yield
    _fid._compression_breaker = _fid._CompressionFidelityBreaker()  # restore default (real home)


def test_low_measured_fidelity_opens_breaker_via_decorator():
    tokeymeter.set_fidelity_circuit_breaker(open_threshold=0.80, close_threshold=0.85,
                                      min_samples=3, window=10, cooldown_s=60)

    @tokeymeter.cache(model="m", tag="wk", compressor=_Compressor(),
                verify_rate=1.0, verify_similarity_fn=lambda o, c: 0.40)
    def ask(prompt):
        return "ok"

    for i in range(5):
        ask(f"please kindly task {i}")     # unique misses -> compress + verify(0.40)
    st = tokeymeter.compression_breaker_state("ask", "wk")
    assert st["state"] == "open", f"breaker should open on measured low fidelity, got {st}"


def test_verify_rate_zero_never_trips_breaker():
    tokeymeter.set_fidelity_circuit_breaker(open_threshold=0.95, close_threshold=0.97,
                                      min_samples=1)

    @tokeymeter.cache(model="m2", tag="wk2", compressor=_Compressor(), verify_rate=0.0)
    def ask(prompt):
        return "ok"

    for i in range(5):
        ask(f"please kindly task {i}")
    st = tokeymeter.compression_breaker_state("ask", "wk2")
    assert st["state"] == "closed"          # no measurement -> no action
    assert st["samples"] == 0
