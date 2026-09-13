"""DecisionRecord — the first-class control-plane primitive."""
import json
import pytest
import tokeymeter
from tokeymeter.storage import MemoryStore
from tokeymeter.compression import StructuralCompressor
from tokeymeter.decision import DecisionRecord, clear_decision_subscribers


@pytest.fixture(autouse=True)
def _reset():
    tokeymeter.set_default_store(MemoryStore())
    tokeymeter.set_default_semantic_cache(None)
    tokeymeter.set_default_redactor(None)
    clear_decision_subscribers()
    yield
    clear_decision_subscribers()


def _collector():
    recs = []
    tokeymeter.on_decision(lambda r: recs.append(r))
    return recs


def test_on_decision_receives_one_record_per_call():
    recs = _collector()
    @tokeymeter.cache(model="m")
    def ask(p): return "a"
    ask("x"); ask("x")            # miss then hit
    assert len(recs) == 2
    assert all(isinstance(r, DecisionRecord) for r in recs)


def test_miss_then_hit_fields():
    recs = _collector()
    @tokeymeter.cache(model="gpt-4o-mini")
    def ask(p): return "answer"
    ask("q"); ask("q")
    miss, hit = recs
    assert miss.decision == "cache_miss" and not miss.cached
    assert miss.cost_usd > 0 and miss.cost_saved_usd == 0
    assert hit.decision == "cache_hit" and hit.cached
    assert hit.cost_saved_usd > 0 and hit.cost_usd == 0
    assert hit.cache_key is not None


def test_high_stakes_decision_recorded():
    recs = _collector()
    @tokeymeter.cache(model="m", high_stakes=True)
    def critical(p): return "ok"
    critical("k")
    assert recs[-1].decision == "high_stakes"
    assert recs[-1].high_stakes and not recs[-1].cached


def test_compression_and_reason_captured():
    recs = _collector()
    @tokeymeter.cache(model="m", compressor=StructuralCompressor())
    def ask(p): return "ok"
    ask("Please kindly note as per our previous discussion " * 3)
    r = recs[-1]
    assert r.compressed and r.compression_ratio is not None
    assert "compressed" in r.reason


def test_provable_reflects_audit_attachment(tmp_path):
    from tokeymeter.audit import AuditLog
    recs = _collector()
    @tokeymeter.cache(model="m")
    def ask(p): return "ok"
    ask("before")                 # no audit attached
    assert recs[-1].provable is False
    assert recs[-1].proof_reference() is None
    audit = AuditLog(path=str(tmp_path/"a.db"),
                     install_secret_path=str(tmp_path/"s"),
                     signing_key_path=str(tmp_path/"k"))
    audit.attach()
    try:
        ask("after")              # audit attached -> provable
        assert recs[-1].provable is True
        assert recs[-1].proof_reference() == recs[-1].cache_key
    finally:
        audit.detach()
    ask("after-detach")
    assert recs[-1].provable is False     # detach restores unprovable


def test_serialization_and_explain():
    recs = _collector()
    @tokeymeter.cache(model="m")
    def ask(p): return "ok"
    ask("x")
    r = recs[-1]
    json.dumps(r.to_dict())               # must not raise
    assert isinstance(r.to_json(), str)
    assert r.decision in r.explain()
    assert r.replayable() is True


def test_subscriber_failure_is_isolated():
    """A raising subscriber must never break the wrapped call."""
    def bad(r): raise RuntimeError("boom")
    tokeymeter.on_decision(bad)
    @tokeymeter.cache(model="m")
    def ask(p): return "ok"
    assert ask("x") == "ok"               # call still succeeds


def test_shadow_decision_labeled():
    recs = _collector()
    @tokeymeter.cache(model="m", shadow=True)
    def ask(p): return "ok"
    ask("x")
    assert recs[-1].decision == "shadow"
    assert recs[-1].shadowed and not recs[-1].replayable()
