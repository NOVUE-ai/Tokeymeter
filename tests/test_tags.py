"""
Tests for workload tagging.

Tags flow through to:
  - savings.jsonl records (per-call)
  - savings_report()["by_tag"] aggregation
  - emitted CacheEvent.tag for observability
"""
import pytest

import tokeymeter
from tokeymeter import events
from tokeymeter.storage import MemoryStore


@pytest.fixture(autouse=True)
def isolated_state():
    tokeymeter.set_default_store(MemoryStore())
    tokeymeter.set_default_semantic_cache(None)
    tokeymeter.set_default_redactor(None)
    tokeymeter.reset_savings()
    events.clear_subscribers()
    yield


def test_tag_flows_to_events():
    seen = []
    events.on_event(seen.append)

    @tokeymeter.cache(tag="support")
    def ask(prompt):
        return "ok"

    ask("hi")
    ask("hi")  # hit

    assert seen[0].tag == "support"
    assert seen[1].tag == "support"


def test_tag_aggregates_in_savings_report():
    @tokeymeter.cache(tag="support", model="gpt-4o-mini")
    def support_bot(prompt):
        return "support response"

    @tokeymeter.cache(tag="rag", model="gpt-4o-mini")
    def rag_bot(prompt):
        return "rag response"

    support_bot("q1")  # miss
    support_bot("q1")  # hit
    support_bot("q2")  # miss
    rag_bot("q3")      # miss

    r = tokeymeter.savings_report()
    assert "support" in r["by_tag"]
    assert "rag" in r["by_tag"]
    assert r["by_tag"]["support"]["calls"] == 3
    assert r["by_tag"]["support"]["exact_hits"] == 1
    assert r["by_tag"]["rag"]["calls"] == 1


def test_untagged_calls_go_into_untagged_bucket():
    @tokeymeter.cache()
    def ask(prompt):
        return "ok"

    ask("hi")
    ask("hi")

    r = tokeymeter.savings_report()
    assert "_untagged" in r["by_tag"]
    assert r["by_tag"]["_untagged"]["calls"] == 2
