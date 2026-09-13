"""Regression tests for P0: strict redaction policy still failed open.

Audit finding (Codex enterprise probe): with SecurityPolicy(require_redaction=True)
and a redactor that raises, the wrapped function still received the raw secret —
the strict policy was enforced only at redactor *resolution* time (a missing
redactor raised) but NOT at *runtime* (a crashing redactor fell open and leaked).

Fix: under strict policy, a redactor that fails (raises OR returns a non-string)
causes a SecurityPolicyError — fail-CLOSED. Under the permissive default, behavior
is unchanged (fail-open). Crucially, when the redactor WORKS, nothing changes, so
prompt/output quality is never affected by this fix.
"""
import pytest

import tokeymeter
from tokeymeter.storage import MemoryStore
from tokeymeter.policy import (
    set_security_policy,
    reset_security_policy,
    SecurityPolicyError,
)
from tokeymeter.privacy import DefaultRedactor


@pytest.fixture(autouse=True)
def _clean():
    reset_security_policy()                 # process-global: isolate each test
    tokeymeter.set_default_store(MemoryStore())
    tokeymeter.reset()
    yield
    reset_security_policy()
    tokeymeter.reset()


def _boom(_text):
    raise RuntimeError("redactor outage")


def test_permissive_policy_still_fails_open_on_redactor_crash():
    """Default behavior is unchanged: a broken redactor must not break the app."""
    seen = {}

    @tokeymeter.cache(model="gpt-4o", redactor=_boom)
    def fn(prompt):
        seen["v"] = prompt
        return "ok"

    # No policy set -> permissive -> fail-open, no exception.
    assert fn("contact alice@example.com") == "ok"
    assert "v" in seen  # the call proceeded


def test_strict_policy_fails_closed_on_redactor_crash():
    """The headline fix: required redaction + crashing redactor => raise, no leak."""
    set_security_policy(require_redaction=True)
    seen = {}

    @tokeymeter.cache(model="gpt-4o", redactor=_boom)
    def fn(prompt):
        seen["v"] = prompt  # must NEVER run with raw PII
        return "ok"

    with pytest.raises(SecurityPolicyError):
        fn("contact alice@example.com")
    assert "v" not in seen  # function never saw the unredacted text


def test_strict_policy_with_working_redactor_is_unaffected():
    """Quality guarantee: when the redactor works, strict policy changes nothing —
    redaction happens, the call succeeds, no exception."""
    set_security_policy(require_redaction=True)
    seen = {}

    @tokeymeter.cache(model="gpt-4o", redactor=DefaultRedactor())
    def fn(prompt):
        seen["v"] = prompt
        return "ok"

    assert fn("contact alice@example.com") == "ok"
    assert "[EMAIL]" in seen["v"]
    assert "alice@example.com" not in seen["v"]


def test_strict_policy_fails_closed_on_nested_input():
    """The fail-closed signal must propagate through nested redaction, not be
    swallowed by the recursive walker."""
    set_security_policy(require_redaction=True)
    seen = {}

    @tokeymeter.cache(model="gpt-4o", redactor=_boom, prompt_arg="messages")
    def chat(messages):
        seen["v"] = messages
        return "ok"

    with pytest.raises(SecurityPolicyError):
        chat(messages=[{"role": "user", "content": "ssn 123-45-6789"}])
    assert "v" not in seen


def test_strict_policy_fails_closed_on_nonstring_redactor_output():
    """A redactor that returns a non-string has not redacted anything; under strict
    policy that is treated as a failure (fail-closed), not a silent passthrough."""
    set_security_policy(require_redaction=True)

    def returns_none(_text):
        return None  # malformed redactor

    seen = {}

    @tokeymeter.cache(model="gpt-4o", redactor=returns_none)
    def fn(prompt):
        seen["v"] = prompt
        return "ok"

    with pytest.raises(SecurityPolicyError):
        fn("contact alice@example.com")
    assert "v" not in seen
