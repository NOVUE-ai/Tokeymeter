"""
Tests for the Prometheus metrics exporter.

Verifies:
  - Render produces valid Prometheus exposition format.
  - Counters increment correctly per (model, hit_type, function) label set.
  - Latency histogram buckets accumulate.
  - Close() unsubscribes cleanly.
  - Multiple collectors don't interfere.
"""
import re

import pytest

import tokeymeter
from tokeymeter import events
from tokeymeter.metrics import PrometheusCollector
from tokeymeter.storage import MemoryStore


@pytest.fixture(autouse=True)
def isolated_state():
    tokeymeter.set_default_store(MemoryStore())
    tokeymeter.set_default_semantic_cache(None)
    tokeymeter.set_default_redactor(None)
    tokeymeter.reset_savings()
    events.clear_subscribers()
    yield
    events.clear_subscribers()


def test_render_produces_valid_prom_format():
    collector = PrometheusCollector()

    @tokeymeter.cache(model="gpt-4o-mini")
    def ask(prompt):
        return "ok"

    ask("hi")
    ask("hi")  # hit

    text = collector.render()

    # Must contain HELP and TYPE lines
    assert "# HELP tokeymeter_events_total" in text
    assert "# TYPE tokeymeter_events_total counter" in text

    # Must have at least one counter line
    counter_lines = [l for l in text.split("\n")
                     if l.startswith("tokeymeter_events_total{")]
    assert len(counter_lines) >= 2  # one for miss, one for hit

    collector.close()


def test_counters_increment_per_hit_type():
    collector = PrometheusCollector()

    @tokeymeter.cache(model="gpt-4o-mini")
    def ask(prompt):
        return "ok"

    # Drive deterministic traffic
    ask("a")     # miss
    ask("a")     # exact hit
    ask("a")     # exact hit
    ask("b")     # miss

    text = collector.render()

    # Extract the events_total counter values
    miss_match = re.search(
        r'tokeymeter_events_total\{[^}]*hit_type="none"[^}]*\}\s+(\d+(?:\.\d+)?)', text
    )
    hit_match = re.search(
        r'tokeymeter_events_total\{[^}]*hit_type="exact"[^}]*\}\s+(\d+(?:\.\d+)?)', text
    )

    assert miss_match is not None, f"no miss counter found in:\n{text}"
    assert hit_match is not None, f"no hit counter found in:\n{text}"
    assert float(miss_match.group(1)) == 2
    assert float(hit_match.group(1)) == 2

    collector.close()


def test_histogram_buckets_accumulate():
    collector = PrometheusCollector()

    @tokeymeter.cache()
    def ask(prompt):
        return "ok"

    for i in range(5):
        ask(f"q-{i}")  # 5 misses

    text = collector.render()

    # Must have histogram lines
    bucket_lines = [l for l in text.split("\n") if "_bucket{" in l]
    assert len(bucket_lines) > 0

    # +Inf bucket should equal total count
    inf_match = re.search(
        r'tokeymeter_latency_seconds_bucket\{[^}]*le="\+Inf"\}\s+(\d+(?:\.\d+)?)', text
    )
    count_match = re.search(
        r'tokeymeter_latency_seconds_count\{[^}]*\}\s+(\d+(?:\.\d+)?)', text
    )
    assert inf_match is not None
    assert count_match is not None
    assert float(inf_match.group(1)) == float(count_match.group(1))

    collector.close()


def test_close_unsubscribes():
    collector = PrometheusCollector()
    initial = events.subscriber_count()
    assert initial >= 1

    collector.close()
    assert events.subscriber_count() == initial - 1

    # After close, new events should not affect the collector
    @tokeymeter.cache()
    def ask(prompt):
        return "ok"

    text_before = collector.render()
    ask("xyz")
    text_after = collector.render()
    assert text_before == text_after


def test_function_name_in_labels():
    collector = PrometheusCollector()

    @tokeymeter.cache()
    def my_special_function(prompt):
        return "ok"

    my_special_function("test")

    text = collector.render()
    assert 'function="my_special_function"' in text

    collector.close()


def test_multiple_collectors_independent():
    c1 = PrometheusCollector(namespace="cache1")
    c2 = PrometheusCollector(namespace="cache2")

    @tokeymeter.cache()
    def ask(prompt):
        return "ok"

    ask("hello")

    t1 = c1.render()
    t2 = c2.render()

    assert "cache1_events_total" in t1
    assert "cache2_events_total" in t2
    # Each should have recorded the event independently
    assert "cache1_events_total" not in t2
    assert "cache2_events_total" not in t1

    c1.close()
    c2.close()
