"""Shared test fixtures.

Tokeymeter exposes a few PROCESS-GLOBAL toggles (event-preview policy, security
policy, the default redactor). A test that flips one and forgets to restore it
silently contaminates later tests — which is exactly how a couple of latent
isolation bugs hid (a test asserting full-preview content passed only because an
earlier test had left the policy on "full"). This autouse fixture resets that
global state to safe defaults around every test, so each test is self-contained
and order-independent.
"""
import pytest

from tokeymeter import decorator as _dec
from tokeymeter.policy import reset_security_policy


@pytest.fixture(autouse=True)
def _reset_global_state():
    def _restore():
        _dec._PREVIEW_POLICY = "hashed"   # the content-blind default
        _dec._default_redactor = None
        reset_security_policy()
    _restore()
    yield
    _restore()


# ── Hypothesis profile (Phase-1 item 3: property-based fuzzing) ──────────
# deadline=None: property tests explore a large input space; a per-example
# wall-clock deadline makes them flaky under CI load (and on slower Windows
# boxes) for no correctness benefit — the invariants, not the timing, are
# what we assert. max_examples is a deliberate CI-time/coverage balance.
try:
    from hypothesis import settings, HealthCheck

    settings.register_profile(
        "tokeymeter",
        max_examples=200,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    settings.load_profile("tokeymeter")
except Exception:
    # hypothesis is a dev-only dependency; its absence must not break
    # collection of the non-property suites.
    pass
