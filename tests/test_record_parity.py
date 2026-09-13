"""Record construction parity (S0-3): one factory, no divergence.

Every CallRecord — whether built by the decorator's _record_and_emit or by an
SDK wrapper's direct emit path — is constructed through savings.build_call_record.
This suite pins that contract so a field added to one path can never be silently
dropped by the other (the exact drift that left the async OpenAI wrapper
emitting records with no pricing_source / principal / key_name).

Pinned:
  1. build_call_record fills EVERY dataclass field (no field can be forgotten).
  2. build_call_record is a pure constructor — no contextvar reads, no
     estimation, no side effects (so callers control resolution).
  3. The decorator path and the async-wrapper path produce records with the
     same field SET, and the async path now carries the fields it used to omit.
"""
import dataclasses

import pytest

import tokeymeter
from tokeymeter import keys as K
from tokeymeter.storage import MemoryStore
from tokeymeter.engines.economics.savings import CallRecord, build_call_record
from tokeymeter.engines.execution.endpoint import endpoint as endpoint_ctx
from tokeymeter.usage import set_reported_usage, set_queue_wait_ms


@pytest.fixture(autouse=True)
def _clean():
    tokeymeter.set_default_store(MemoryStore())
    tokeymeter.set_in_memory_savings(True)
    tokeymeter.reset_savings()
    K.clear_keys()
    yield
    K.clear_keys()
    tokeymeter.reset_savings()


def _records():
    from tokeymeter import savings as sv
    return list(sv._tracker._iter_records())


# ── the factory contract ────────────────────────────────────────────────

def test_factory_covers_every_dataclass_field():
    # Guard: if a field is added to CallRecord but not to build_call_record's
    # signature, this fails — forcing the factory to stay complete. We check
    # SIGNATURE COVERAGE (not non-null values — several fields are legitimately
    # None, e.g. hit_type on a miss, tag when unset).
    import inspect
    record_fields = {f.name for f in dataclasses.fields(CallRecord)}
    factory_params = set(inspect.signature(build_call_record).parameters)
    # timestamp is set internally by the factory, not a caller-supplied param
    missing = record_fields - factory_params - {"timestamp"}
    assert not missing, (
        f"CallRecord fields not accepted by build_call_record: {missing}. "
        "Add them to the factory so every record path picks them up.")


def test_factory_is_pure_no_contextvar_reads():
    # Bind contextvars that the DECORATOR would read; the factory must ignore
    # them — it only uses what it is explicitly passed.
    with endpoint_ctx("should-be-ignored"):
        set_queue_wait_ms(999.0)
        r = build_call_record(
            model="m", hit=False, hit_type=None,
            input_tokens=1, output_tokens=1, estimated_cost=0.0, latency_ms=1.0,
        )
    assert r.endpoint_identity is None    # not read from contextvar
    assert r.queue_wait_ms is None        # not read from contextvar


def test_factory_defaults_are_none_and_zero():
    r = build_call_record(
        model="m", hit=True, hit_type="exact",
        input_tokens=0, output_tokens=0, estimated_cost=0.0, latency_ms=0.1,
    )
    assert r.pricing_source is None
    assert r.principal is None
    assert r.key_name is None
    assert r.endpoint_identity is None
    assert r.queue_wait_ms is None
    assert r.tokens_saved_via_compression == 0


# ── decorator path completeness (regression guard) ──────────────────────

def test_decorator_record_has_full_field_set():
    tokeymeter.register_key("prod", "sk-abcdef123456", monthly_cap_usd=100)
    with tokeymeter.key("prod"):
        with tokeymeter.principal("person:priya"):
            with endpoint_ctx("vllm-pool"):
                @tokeymeter.cache(model="gpt-4o")
                def ask(p):
                    set_reported_usage(100, 50)
                    set_queue_wait_ms(12.0)
                    return "r"
                ask("hello")
    rec = _records()[-1]
    # every attribution/provenance field present on a decorator-built record
    assert rec["pricing_source"] is not None
    assert rec["principal"] == "person:priya"
    assert rec["key_name"] == "prod"
    assert rec["token_source"] == "reported"
    assert rec["endpoint_identity"] == "vllm-pool"
    assert rec["queue_wait_ms"] == 12.0


# ── async-wrapper path completeness (the gap S0-3 closes) ───────────────

class _Usage:
    prompt_tokens = 200
    completion_tokens = 80


class _Resp:
    usage = _Usage()


async def test_async_wrapper_record_now_carries_provenance_and_identity():
    # Drive the async OpenAI wrapper's _emit_record directly with a fake
    # response, under bound identity/endpoint, and assert the previously-missing
    # fields are now present.
    from tokeymeter.engines.execution.integrations.openai_async import (
        _AsyncCachedCompletions,
    )

    # a minimal stand-in inner client the wrapper wraps
    class _Inner:
        pass

    wrapper = _AsyncCachedCompletions.__new__(_AsyncCachedCompletions)
    wrapper._shadow = False
    wrapper._tag = None

    tokeymeter.register_key("prod", "sk-abcdef123456", monthly_cap_usd=100)
    with tokeymeter.key("prod"):
        with tokeymeter.principal("agent:evals"):
            with endpoint_ctx("vllm-async-pool"):
                import time as _t
                wrapper._emit_record(_Resp(), "gpt-4o", False, None, _t.perf_counter())

    rec = _records()[-1]
    assert rec["pricing_source"] is not None      # was silently None before S0-3
    assert rec["principal"] == "agent:evals"       # was silently None before S0-3
    assert rec["key_name"] == "prod"               # was silently None before S0-3
    assert rec["token_source"] == "reported"
    assert rec["endpoint_identity"] == "vllm-async-pool"
    assert rec["input_tokens"] == 200 and rec["output_tokens"] == 80


async def test_both_paths_produce_identical_field_sets():
    # Decorator path
    @tokeymeter.cache(model="gpt-4o")
    def ask(p):
        set_reported_usage(10, 20)
        return "r"
    ask("via-decorator")
    decorator_keys = set(_records()[-1].keys())

    # Async-wrapper path
    from tokeymeter.engines.execution.integrations.openai_async import (
        _AsyncCachedCompletions,
    )
    wrapper = _AsyncCachedCompletions.__new__(_AsyncCachedCompletions)
    wrapper._shadow = False
    wrapper._tag = None
    import time as _t
    wrapper._emit_record(_Resp(), "gpt-4o", False, None, _t.perf_counter())
    wrapper_keys = set(_records()[-1].keys())

    assert decorator_keys == wrapper_keys, (
        "record field sets diverged between paths — the shared factory should "
        f"make this impossible. decorator-only={decorator_keys - wrapper_keys}, "
        f"wrapper-only={wrapper_keys - decorator_keys}")
