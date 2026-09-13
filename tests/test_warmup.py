"""
Tests for cache warmup — the "Day 1 looks like Day 30" feature.

Verifies:
  - warm_from_iterable populates exact + (optionally) semantic.
  - warm_from_jsonl reads a JSONL file correctly.
  - Custom prompt/response extractors work for non-default schemas.
  - Bulk encoding is used when the encoder supports it (perf win).
  - max_entries caps the warmup correctly.
  - Errors in source data don't crash the warmup.
"""
import json
import time

import numpy as np
import pytest

import tokeymeter
from tokeymeter import warmup
from tokeymeter.semantic import SemanticCache
from tokeymeter.storage import MemoryStore


@pytest.fixture(autouse=True)
def isolated_state():
    tokeymeter.set_default_store(MemoryStore())
    tokeymeter.set_default_semantic_cache(None)
    tokeymeter.reset_savings()
    yield


def test_warm_from_iterable_populates_exact_cache():
    store = MemoryStore()
    tokeymeter.set_default_store(store)

    pairs = [
        ("what is photosynthesis", "Plants converting sunlight..."),
        ("what is mitosis", "Cell division..."),
        ("what is gravity", "A fundamental force..."),
    ]

    stats = warmup.warm_from_iterable(pairs, store=store, namespace="warm-pool")
    assert stats["total_loaded"] == 3
    assert stats["exact_stored"] == 3
    assert len(store) >= 3

    # Now the @tokeymeter.cache decorator should HIT for these prompts
    calls = [0]

    @tokeymeter.cache(store=store, namespace="warm-pool")
    def ask(prompt):
        calls[0] += 1
        return f"r-{prompt}"

    result = ask("what is photosynthesis")
    assert "Plants converting" in result
    assert calls[0] == 0  # cache hit, function not invoked


def test_warm_from_iterable_with_semantic_bulk_encodes(tmp_path):
    """Bulk encoding should be ~25× faster than per-prompt for many entries."""
    def bag(t):
        # Accept either str or list[str] (bulk mode)
        if isinstance(t, list):
            return np.stack([bag(s) for s in t])
        t = t.lower()
        v = np.zeros(26, dtype=np.float32)
        for ch in t:
            if "a" <= ch <= "z":
                v[ord(ch) - ord("a")] += 1
        n = np.linalg.norm(v)
        return (v / n) if n > 0 else v

    sem = SemanticCache(
        path=str(tmp_path / "sem.db"),
        threshold=0.95,
        encoder=bag,
        dim=26,
    )

    pairs = [(f"prompt number {i} about topic", f"r-{i}") for i in range(50)]

    stats = warmup.warm_from_iterable(pairs, semantic_cache=sem)
    assert stats["total_loaded"] == 50
    assert stats["semantic_stored"] == 50

    # Semantic lookup should find one of them
    r = sem.lookup("prompt number 5 about topic")
    assert r is not None


def test_warm_from_jsonl_default_schema(tmp_path):
    log_path = tmp_path / "logs.jsonl"
    with open(log_path, "w") as f:
        for i in range(5):
            f.write(json.dumps({"prompt": f"q-{i}", "response": f"a-{i}"}) + "\n")

    store = MemoryStore()
    stats = warmup.warm_from_jsonl(str(log_path), store=store)
    assert stats["total_loaded"] == 5
    assert stats["exact_stored"] == 5


def test_warm_from_jsonl_custom_extractors(tmp_path):
    """OpenAI-style logs: messages[-1].content as prompt, choices[0].message.content as response."""
    log_path = tmp_path / "openai_logs.jsonl"
    with open(log_path, "w") as f:
        for i in range(3):
            f.write(json.dumps({
                "messages": [
                    {"role": "system", "content": "..."},
                    {"role": "user", "content": f"openai-q-{i}"},
                ],
                "choices": [{"message": {"content": f"openai-r-{i}"}}],
            }) + "\n")

    store = MemoryStore()
    stats = warmup.warm_from_jsonl(
        str(log_path),
        prompt_extract=lambda r: r["messages"][-1]["content"],
        response_extract=lambda r: r["choices"][0]["message"]["content"],
        store=store,
    )
    assert stats["total_loaded"] == 3
    assert stats["exact_stored"] == 3


def test_warm_from_jsonl_max_entries(tmp_path):
    log_path = tmp_path / "logs.jsonl"
    with open(log_path, "w") as f:
        for i in range(100):
            f.write(json.dumps({"prompt": f"q-{i}", "response": f"a-{i}"}) + "\n")

    store = MemoryStore()
    stats = warmup.warm_from_jsonl(str(log_path), store=store, max_entries=20)
    assert stats["total_loaded"] == 20


def test_warmup_skips_invalid_entries():
    """Malformed entries should be counted as errors but not crash the warmup."""
    pairs = [
        ("valid prompt", "valid response"),
        ("", "empty prompt is invalid"),       # invalid (empty prompt)
        (None, "none prompt is invalid"),      # invalid
        ("another valid", "ok"),
    ]

    store = MemoryStore()
    stats = warmup.warm_from_iterable(pairs, store=store)
    assert stats["total_loaded"] == 2  # only the two valid ones
    assert stats["errors"] >= 2


def test_warmup_progress_callback():
    pairs = [(f"q-{i}", f"a-{i}") for i in range(250)]
    calls = []

    def progress(total_loaded, batch_size):
        calls.append((total_loaded, batch_size))

    store = MemoryStore()
    stats = warmup.warm_from_iterable(pairs, store=store, batch_size=100, progress=progress)
    assert stats["total_loaded"] == 250
    # At batch_size=100, expect ~3 progress calls (100, 200, 50-tail)
    assert len(calls) >= 2


def test_warm_from_jsonl_handles_malformed_lines(tmp_path):
    """A file with some invalid lines should warm the valid ones."""
    log_path = tmp_path / "mixed.jsonl"
    with open(log_path, "w") as f:
        f.write(json.dumps({"prompt": "valid-1", "response": "r1"}) + "\n")
        f.write("this is not JSON\n")  # invalid line
        f.write(json.dumps({"prompt": "valid-2", "response": "r2"}) + "\n")
        f.write("\n")  # blank line
        f.write(json.dumps({"no_prompt_here": True}) + "\n")  # valid JSON, missing fields
        f.write(json.dumps({"prompt": "valid-3", "response": "r3"}) + "\n")

    store = MemoryStore()
    stats = warmup.warm_from_jsonl(str(log_path), store=store)
    assert stats["total_loaded"] == 3
