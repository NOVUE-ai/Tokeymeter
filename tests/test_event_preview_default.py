"""Regression tests for P1: event previews leaked prompt content by default,
and for the enterprise_defaults() safe-posture switch.

Audit finding (Codex enterprise probe): _PREVIEW_POLICY defaulted to "full", so
observability subscribers received raw prompt text (first ~200 chars) by default
— contradicting the documented content-blind/local-first posture.

Fix: the default is now "hashed" (a non-reversible correlation token, no content);
"full" is opt-in for local debugging. enterprise_defaults() bundles the safe
posture (hashed previews + required redaction with a redactor ensured + runtime
guards) in one idempotent, immediately-satisfiable call.

(The autouse fixture in conftest.py resets global state, so these are isolated.)
"""
import tokeymeter
from tokeymeter import decorator as _dec
from tokeymeter import events
from tokeymeter.storage import MemoryStore
from tokeymeter.policy import get_security_policy


def _fresh():
    tokeymeter.set_default_store(MemoryStore())
    tokeymeter.reset()


def test_default_preview_is_hashed_no_content_leak():
    """Headline fix: by default, event previews carry NO raw prompt content."""
    _fresh()
    assert _dec._PREVIEW_POLICY == "hashed"
    seen = []
    events.on_event(seen.append)

    @tokeymeter.cache(model="gpt-4o")
    def ask(prompt):
        return "ok"

    secret = "the user's confidential question about Project Falcon"
    ask(secret)
    preview = seen[-1].prompt_preview or ""
    assert secret[:20] not in preview          # no content
    assert preview.startswith("sha256:")       # just a correlation token


def test_full_preview_still_available_opt_in():
    """Opt-in 'full' mode still works for local debugging."""
    _fresh()
    _dec.set_event_preview_policy("full")
    seen = []
    events.on_event(seen.append)

    @tokeymeter.cache(model="gpt-4o")
    def ask(prompt):
        return "ok"

    ask("visible prompt text")
    assert "visible prompt text" in (seen[-1].prompt_preview or "")


def test_enterprise_defaults_sets_safe_posture():
    """enterprise_defaults() flips the documented safe posture in one call."""
    _fresh()
    summary = tokeymeter.enterprise_defaults()
    assert _dec._PREVIEW_POLICY == "hashed"
    assert _dec._RUNTIME_GUARDS is True
    assert get_security_policy().require_redaction is True
    assert _dec._default_redactor is not None  # require_redaction is satisfiable
    assert summary["redactor_configured"] is True


def test_enterprise_defaults_is_immediately_satisfiable():
    """After enterprise_defaults(), a normal call must succeed (not fail closed
    for lack of a redactor) AND emit a hashed, PII-free preview."""
    _fresh()
    tokeymeter.enterprise_defaults()
    seen = []
    events.on_event(seen.append)

    @tokeymeter.cache(model="gpt-4o")
    def ask(prompt):
        return "ok"

    # must not raise (a redactor was ensured), and must redact + hash
    assert ask("email alice@example.com") == "ok"
    preview = seen[-1].prompt_preview or ""
    assert "alice@example.com" not in preview
    assert preview.startswith("sha256:")


def test_enterprise_defaults_respects_supplied_redactor():
    """A caller-supplied redactor is honored over the built-in default."""
    _fresh()
    marker = {"used": False}

    def my_redactor(text):
        marker["used"] = True
        return "[SCRUBBED]"

    tokeymeter.enterprise_defaults(redactor=my_redactor)

    @tokeymeter.cache(model="gpt-4o")
    def ask(prompt):
        return "ok"

    ask("anything")
    assert marker["used"] is True


def test_enterprise_defaults_is_idempotent():
    """Calling it twice yields the same posture (no drift)."""
    _fresh()
    a = tokeymeter.enterprise_defaults()
    b = tokeymeter.enterprise_defaults()
    assert a["event_preview_policy"] == b["event_preview_policy"] == "hashed"
    assert a["security_policy"] == b["security_policy"]
