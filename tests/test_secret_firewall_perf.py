"""
P0.4 — secret-firewall performance guard.

A 5 MB paste must never time out the scanner. These tests prove the scan is
latency-bounded on large and adversarial inputs, that detection is preserved
(including a secret straddling a chunk boundary), that the cheap prefilter is
sound, that posture governs an incomplete scan (block vs proceed), and that no
raw secret ever leaks into findings, descriptors, or errors.

    PYTHONPATH=. python -m pytest tests/test_secret_firewall_perf.py -q
"""
import time

import pytest

from tokeymeter.content.secrets import (
    SecretScanner, SecretFirewall, SecretPolicy, SecretBlocked,
    _CHUNK_BYTES,
)

TARGET_MS = 1500   # a single scan must return well under this on a 5 MB input
KEY = "sk-ant-api03-" + "Xy9zAb3Qw7Rt2Uv6" * 3   # a realistic-looking Anthropic key


def _scan_ms(scanner, text, **kw):
    t0 = time.perf_counter()
    r = scanner.scan(text, **kw)
    return (time.perf_counter() - t0) * 1000, r


def test_large_base64_blob_is_bounded():
    sc = SecretScanner()
    blob = (("A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8S9t0" * 4) * 32000)[:5_000_000]
    ms, r = _scan_ms(sc, blob)
    assert ms < TARGET_MS, f"5MB base64 took {ms:.0f}ms"
    assert r.scanned is True


def test_redteam_continuous_digit_bomb_is_bounded():
    # a 5MB run of digits is the classic credit-card backtracking bomb
    sc = SecretScanner()
    ms, r = _scan_ms(sc, "9" * 5_000_000)
    assert ms < TARGET_MS, f"digit bomb took {ms:.0f}ms"


def test_mixed_burst_stays_fast():
    sc = SecretScanner()
    t0 = time.perf_counter()
    for _ in range(200):
        sc.scan("a normal prompt about cats and dogs. " * 500)   # ~18 KB each
    per = (time.perf_counter() - t0) * 1000 / 200
    assert per < 25, f"{per:.1f}ms per medium scan"


def test_detection_preserved_in_large_input():
    # a real key buried in the middle of 5 MB must still be caught, fast
    sc = SecretScanner()
    big = ("the quick brown fox. " * 60000)[:2_500_000] + " " + KEY + " " + \
          ("more text here. " * 60000)
    big = big[:5_000_000]
    ms, r = _scan_ms(sc, big)
    assert ms < TARGET_MS
    assert "ANTHROPIC_API_KEY" in {f.type for f in r.findings}


def test_secret_straddling_a_chunk_boundary_is_caught():
    # place the key so it spans the boundary between chunk 1 and chunk 2;
    # the chunk overlap must still catch it
    sc = SecretScanner()
    pad = "x" * (_CHUNK_BYTES - 5)        # key begins 5 chars before the boundary
    text = pad + KEY + "y" * 1000
    r = sc.scan(text)
    assert "ANTHROPIC_API_KEY" in {f.type for f in r.findings}


def test_prefilter_is_sound_skips_when_literal_absent():
    # a large input with no secret markers → fast, zero findings
    sc = SecretScanner()
    ms, r = _scan_ms(sc, "lorem ipsum dolor sit amet. " * 200000)   # ~5.4 MB, no secrets
    assert ms < TARGET_MS
    assert len(r.findings) == 0


def test_small_prompt_fast_path_unchanged():
    sc = SecretScanner()
    r = sc.scan("Summarize this. My key is " + KEY)
    assert r.complete is True
    assert "ANTHROPIC_API_KEY" in {f.type for f in r.findings}


class _ZeroBudgetScanner(SecretScanner):
    """Forces an incomplete scan on any large input (budget exhausted immediately)."""
    def scan(self, text, *, time_budget_s=None):
        return super().scan(text, time_budget_s=0.0)


def test_posture_block_fails_closed_on_incomplete_scan():
    fw = SecretFirewall(scanner=_ZeroBudgetScanner(), policy=SecretPolicy(fail_closed=True))
    with pytest.raises(SecretBlocked) as ei:
        fw.enforce("x" * 1_000_000)
    assert "SCAN_INCOMPLETE" in {d["type"] for d in ei.value.descriptors}


def test_posture_measure_proceeds_on_incomplete_scan():
    fw = SecretFirewall(scanner=_ZeroBudgetScanner(), policy=SecretPolicy(fail_closed=False))
    res = fw.enforce("x" * 1_000_000)        # fail-open: no raise
    assert res.safe_text is not None


def test_input_size_budget_is_deterministic():
    # The byte budget caps work identically on every machine (unlike a time
    # budget): a 5 MB input with a 1 MB budget scans ~1 MB, marks incomplete,
    # and returns the same bytes_scanned every run.
    sc = SecretScanner()
    r1 = sc.scan("x" * 5_000_000, max_bytes=1_000_000)
    r2 = sc.scan("x" * 5_000_000, max_bytes=1_000_000)
    assert r1.bytes_scanned <= 1_000_000 + _CHUNK_BYTES
    assert r1.complete is False
    assert r1.bytes_scanned == r2.bytes_scanned        # deterministic across runs


def test_oversized_input_under_default_budget_marks_incomplete():
    # an input beyond the default 16 MB budget is bounded and flagged incomplete
    from tokeymeter.content.secrets import _MAX_SCAN_BYTES
    sc = SecretScanner()
    r = sc.scan("a" * (_MAX_SCAN_BYTES + 2_000_000))
    assert r.complete is False
    assert r.bytes_scanned <= _MAX_SCAN_BYTES + _CHUNK_BYTES


def test_no_raw_secret_in_findings_or_descriptors_or_errors():
    text = "here is my key " + KEY + " please use it"
    sc = SecretScanner()
    r = sc.scan(text)
    # findings carry type/span/confidence only — never the value
    for f in r.findings:
        assert KEY not in str(f.__dict__)
    assert r.error is None or KEY not in r.error
    # the block descriptors are content-blind too
    fw = SecretFirewall()
    with pytest.raises(SecretBlocked) as ei:
        fw.enforce(text)
    for d in ei.value.descriptors:
        assert KEY not in str(d)
