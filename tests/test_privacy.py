"""
Tests for the PII redactor and its integration with the cache.
"""
import pytest

import tokeymeter
from tokeymeter import events
from tokeymeter.privacy import DefaultRedactor, default_redactor, redact
from tokeymeter.storage import MemoryStore


@pytest.fixture(autouse=True)
def isolated_state():
    tokeymeter.set_default_store(MemoryStore())
    tokeymeter.set_default_semantic_cache(None)
    tokeymeter.set_default_redactor(None)
    tokeymeter.reset_savings()
    events.clear_subscribers()
    yield


# ---------- Unit tests for the default redactor ----------

def test_redacts_email():
    assert "alice@example.com" not in redact("contact alice@example.com please")
    assert "[EMAIL]" in redact("contact alice@example.com please")


def test_redacts_ssn():
    assert redact("SSN is 123-45-6789") == "SSN is [SSN]"


def test_redacts_credit_card_with_luhn():
    # Valid Visa test number
    assert "[CC]" in redact("card: 4111 1111 1111 1111")
    # Invalid number (fails Luhn) — should NOT redact
    assert "[CC]" not in redact("not a card: 4111 1111 1111 1112")


def test_redacts_e164_phone():
    assert redact("call +14155551234") == "call [PHONE]"


def test_redacts_ipv4():
    assert "[IP]" in redact("server at 192.168.1.42 is down")
    assert "[IP]" in redact("ping 8.8.8.8")


def test_redacts_api_keys():
    assert "[APIKEY]" in redact("key sk-proj-aaaaaaaaaaaaaaaaaaaa1234")
    assert "[APIKEY]" in redact("key sk-ant-api03-aaaaaaaaaaaaaaaaaaaaa")
    assert "[APIKEY]" in redact("key AKIAIOSFODNN7EXAMPLE")


def test_does_not_over_redact_clean_text():
    clean = "Please explain quantum entanglement in simple terms."
    assert redact(clean) == clean


def test_custom_patterns():
    red = DefaultRedactor(custom_patterns=[
        (r"\bORDER-\d{6}\b", "[ORDER]"),
    ])
    assert red("see ORDER-123456 for details") == "see [ORDER] for details"


def test_redactor_handles_non_string():
    # If something somehow passes a non-string to redact(), it should not crash
    assert default_redactor(None) is None  # type: ignore[arg-type]
    assert default_redactor("") == ""


# ---------- Integration with @tokeymeter.cache ----------

def test_redacted_prompts_with_different_pii_share_cache_key():
    """Two prompts that differ only in email should hit the same cache entry."""
    calls = [0]

    @tokeymeter.cache(redactor=default_redactor)
    def ask(prompt):
        calls[0] += 1
        return f"r-{calls[0]}"

    r1 = ask("send email to alice@example.com asap")  # miss
    r2 = ask("send email to bob@example.com asap")    # same key after redaction
    assert r1 == r2
    assert calls[0] == 1


def test_redactor_does_not_break_normal_caching():
    calls = [0]

    @tokeymeter.cache(redactor=default_redactor)
    def ask(prompt):
        calls[0] += 1
        return "ok"

    ask("hello world")
    ask("hello world")  # exact hit
    assert calls[0] == 1


def test_event_preview_is_redacted():
    # Verifies that even with the opt-in "full" preview, redaction scrubs PII.
    from tokeymeter import decorator as _dec
    _dec.set_event_preview_policy("full")
    seen = []
    events.on_event(seen.append)

    @tokeymeter.cache(redactor=default_redactor)
    def ask(prompt):
        return "ok"

    ask("contact alice@example.com")

    assert "alice@example.com" not in seen[0].prompt_preview
    assert "[EMAIL]" in seen[0].prompt_preview


def test_broken_redactor_fails_open():
    def broken(text):
        raise RuntimeError("redactor crashed")

    calls = [0]

    @tokeymeter.cache(redactor=broken)
    def ask(prompt):
        calls[0] += 1
        return "ok"

    # Should not raise — the call still works, just without redaction
    assert ask("hello") == "ok"
    assert ask("hello") == "ok"  # cache still works (key derived from un-redacted)
    assert calls[0] == 1


def test_redact_response_prevents_cached_pii_replay():
    redactor = DefaultRedactor()
    calls = [0]

    @tokeymeter.cache(model="m", redactor=redactor, redact_response=True)
    def ask(prompt):
        calls[0] += 1
        return {"answer": "email alice@example.com"}

    first = ask("summarize user")
    second = ask("summarize user")

    assert calls[0] == 1
    assert first == {"answer": "email [EMAIL]"}
    assert second == {"answer": "email [EMAIL]"}


@pytest.mark.asyncio
async def test_cache_stream_redact_response():
    redactor = DefaultRedactor()

    @tokeymeter.cache_stream(model="m", redactor=redactor, redact_response=True)
    async def stream(prompt):
        yield "token alice@example.com"

    out = [c async for c in stream("hello")]
    assert out == ["token [EMAIL]"]


def test_default_redactor_can_be_disabled_at_app_level():
    """set_default_redactor(None) reverts to no redaction even with shared deco."""
    tokeymeter.set_default_redactor(default_redactor)

    calls = [0]

    @tokeymeter.cache()
    def ask(prompt):
        calls[0] += 1
        return f"r-{calls[0]}"

    r1 = ask("call +14155551234")
    r2 = ask("call +14155556789")  # different phone → same redacted key → hit
    assert calls[0] == 1
    assert r1 == r2

    # Now disable globally
    tokeymeter.set_default_redactor(None)

    r3 = ask("call +14155551234")  # exact key now differs from redacted
    # Should miss because the previous calls were stored under the redacted key
    assert calls[0] == 2


# --- Regression tests for H4: redaction recall + compliance honesty ---

def test_h4_expanded_recall():
    """Spaced SSN, US phone formats, and Aadhaar are now redacted."""
    assert "[SSN]" in redact("SSN 123 45 6789")
    assert "[PHONE]" in redact("call (555) 123-4567")
    assert "[PHONE]" in redact("call 555-123-4567")
    assert "[AADHAAR]" in redact("Aadhaar 1234 5678 9012")
    # Must NOT mangle a 16-digit credit card as Aadhaar:
    assert "[CC]" in redact("card 4111 1111 1111 1111")
    assert "[AADHAAR]" not in redact("card 4111 1111 1111 1111")
