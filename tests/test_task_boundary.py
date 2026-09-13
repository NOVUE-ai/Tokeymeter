"""Task boundary (S4-1) — the unit of accounting becomes the task, not the call.

Pinned here, because each of these was a real decision:

  PRE-FLIGHT, NOT POST-MORTEM. Limits are evaluated BEFORE a call proceeds,
    against state from earlier calls in the same task. Checking afterwards
    means the spend already happened.
  A LOOP SERVED FROM CACHE IS STILL A LOOP. Repeat counting includes cache
    hits — an agent re-asking the same question is stuck whether or not the
    answer came from cache. Spend, by contrast, only accrues on executed calls.
  CONTENT-BLIND. Loop detection compares SHA-256 digests of the cache key. No
    prompt is read, and the record gains two fields while carrying zero
    content.
  FAIL-OPEN BY DEFAULT. `enforce=False` records the breach and lets the call
    through, so a team can measure what a ceiling WOULD have stopped before it
    stops anything. Adoption is risk-free; the ceiling is a separate decision.
  CONCURRENCY IS NOT UNIFORM. asyncio inherits the binding; plain threads do
    not. That asymmetry is a correctness trap (silent non-enforcement), so
    both paths are pinned.
"""
import asyncio
import concurrent.futures as cf
import json
import math

import pytest

import tokeymeter
from tokeymeter.storage import MemoryStore
from tokeymeter.engines.execution import task as T
from tokeymeter.engines.economics.usage import set_reported_usage
import tokeymeter.engines.economics.savings as sv


@pytest.fixture(autouse=True)
def _clean():
    tokeymeter.set_default_store(MemoryStore())
    tokeymeter.set_in_memory_savings(True)
    tokeymeter.reset_savings()
    tokeymeter.clear_registered_pricing()
    tokeymeter.register_pricing("m", input_per_1m=2.5, output_per_1m=10.0)
    yield
    tokeymeter.set_in_memory_savings(False)
    tokeymeter.reset_savings()
    tokeymeter.clear_registered_pricing()


def _step(model="m", tag=None):
    @tokeymeter.cache(model=model, tag=tag)
    def step(p):
        set_reported_usage(3000, 800)          # $0.0155 per executed call
        return "response"
    return step


def _records():
    return list(sv._tracker._iter_records())


# ── binding and stamping ────────────────────────────────────────────────

def test_task_id_stamped_on_every_record():
    step = _step()
    with tokeymeter.task("ticket-4f2a"):
        step("a")
        step("b")
    assert {r["task_id"] for r in _records()} == {"ticket-4f2a"}


def test_no_task_leaves_fields_unset():
    """Backward compatible: calls outside a boundary carry task_id=None and
    group as unattributed, exactly as principal=None already does."""
    step = _step()
    step("x")
    r = _records()[0]
    assert r["task_id"] is None
    assert r["prompt_fingerprint"] is not None   # fingerprint is always useful


def test_binding_restored_after_exception():
    step = _step()
    with pytest.raises(ValueError):
        with tokeymeter.task("t-err"):
            step("a")
            raise ValueError("boom")
    assert tokeymeter.current_task_id() is None


def test_nested_tasks_restore_outer():
    with tokeymeter.task("outer"):
        assert tokeymeter.current_task_id() == "outer"
        with tokeymeter.task("inner"):
            assert tokeymeter.current_task_id() == "inner"
        assert tokeymeter.current_task_id() == "outer"
    assert tokeymeter.current_task_id() is None


# ── fingerprint ─────────────────────────────────────────────────────────

def test_identical_calls_share_a_fingerprint():
    step = _step()
    with tokeymeter.task("t"):
        step("same")
        step("same")
    assert len({r["prompt_fingerprint"] for r in _records()}) == 1


def test_different_calls_differ():
    step = _step()
    with tokeymeter.task("t"):
        step("one")
        step("two")
    assert len({r["prompt_fingerprint"] for r in _records()}) == 2


def test_fingerprint_is_a_non_reversible_digest():
    fp = T.fingerprint_of("some-cache-key")
    assert fp.startswith("sha256:") and len(fp) == len("sha256:") + 12
    assert T.fingerprint_of(None) is None
    assert T.fingerprint_of("") is None


# ── content-blindness ───────────────────────────────────────────────────

def test_record_carries_zero_content():
    step = _step()
    with tokeymeter.task("phi-case"):
        step("patient SSN 123-45-6789 confidential diagnosis text")
    blob = json.dumps(_records()[0])
    assert "123-45-6789" not in blob
    assert "confidential" not in blob
    assert "diagnosis" not in blob


# ── enforcement: the ceiling ────────────────────────────────────────────

def test_loop_halts_on_repeated_fingerprint():
    step = _step()
    with pytest.raises(tokeymeter.TaskLoopDetected) as ei:
        with tokeymeter.task("t", max_repeats=4, enforce=True):
            for _ in range(25):
                step("the identical failing call")
    assert ei.value.limit == "max_repeats"
    assert ei.value.allowed == 4


def test_loop_detected_even_when_served_from_cache():
    """The repeat that trips the limit is a cache HIT — an agent re-asking the
    same question is stuck whether or not the answer was cached."""
    calls = {"n": 0}

    @tokeymeter.cache(model="m")
    def step(p):
        calls["n"] += 1
        set_reported_usage(3000, 800)
        return "r"

    with pytest.raises(tokeymeter.TaskLoopDetected):
        with tokeymeter.task("t", max_repeats=3, enforce=True):
            for _ in range(10):
                step("same")
    assert calls["n"] == 1          # only the first was a real call
    assert sum(1 for r in _records() if r["hit"]) >= 1


def test_envelope_halts_on_spend():
    step = _step()
    with pytest.raises(tokeymeter.TaskEnvelopeExceeded) as ei:
        with tokeymeter.task("t", envelope=0.05, enforce=True):
            for i in range(50):
                step(f"distinct-{i}")
    assert ei.value.limit == "envelope"
    # `observed` is ACTUAL accumulated spend, always below the envelope now
    # that the worst case is held before the call rather than after it.
    assert 0 < ei.value.observed <= 0.05


def test_max_calls_halts():
    step = _step()
    with pytest.raises(tokeymeter.TaskCallLimitExceeded) as ei:
        with tokeymeter.task("t", max_calls=5, enforce=True):
            for i in range(20):
                step(f"d-{i}")
    assert ei.value.allowed == 5


def test_all_limit_types_share_a_base_class():
    """A caller catches one thing and handles any ceiling breach uniformly."""
    step = _step()
    with pytest.raises(tokeymeter.TaskLimitExceeded):
        with tokeymeter.task("t", max_calls=2, enforce=True):
            for i in range(10):
                step(f"d-{i}")


def test_cache_hits_do_not_add_spend():
    """A hit cost nothing upstream, so it must not consume the envelope."""
    step = _step()
    with tokeymeter.task("t", envelope=100.0) as st:
        step("same")
        for _ in range(20):
            step("same")           # all hits
    snap = st.snapshot()
    assert snap["calls"] == 21
    assert snap["spend_usd"] == pytest.approx(0.0155, abs=1e-4)


# ── observe mode ────────────────────────────────────────────────────────

def test_observe_mode_records_breach_without_raising():
    step = _step()
    with tokeymeter.task("t", envelope=0.05, enforce=False) as st:
        for i in range(20):
            step(f"d-{i}")          # must NOT raise
    snap = st.snapshot()
    assert snap["halted_reason"] == "envelope"
    assert snap["spend_usd"] > 0.05
    assert snap["calls"] == 20


def test_enforce_defaults_to_false():
    """Adopting the boundary can never take down an agent that was working."""
    step = _step()
    with tokeymeter.task("t", envelope=0.001) as st:
        step("a")
        step("b")
    assert st.snapshot()["enforce"] is False


# ── concurrency ─────────────────────────────────────────────────────────

def test_asyncio_inherits_the_binding():
    @tokeymeter.cache(model="m")
    async def astep(p):
        set_reported_usage(3000, 800)
        return "r"

    async def main():
        with tokeymeter.task("async-t", envelope=100.0) as st:
            await asyncio.gather(*[astep(f"a-{i}") for i in range(20)])
            return st.snapshot()
    snap = asyncio.run(main())
    assert snap["calls"] == 20


def test_threads_need_bind_task_and_then_share_one_envelope():
    step = _step()
    with tokeymeter.task("threaded", envelope=100.0) as t:
        def work(i):
            with tokeymeter.bind_task(t):
                return step(f"w-{i}")
        with cf.ThreadPoolExecutor(8) as ex:
            list(ex.map(work, range(40)))
    assert t.snapshot()["calls"] == 40


def test_threads_without_bind_are_not_counted():
    """Pins the documented asymmetry so it can never regress into a silent
    non-enforcement bug: a plain worker sees no task at all."""
    with tokeymeter.task("t"):
        with cf.ThreadPoolExecutor(2) as ex:
            seen = list(ex.map(lambda i: tokeymeter.current_task_id(), range(2)))
    assert seen == [None, None]


def test_enforcement_holds_across_a_thread_pool():
    step = _step()
    halted = {"n": 0}
    with tokeymeter.task("t", envelope=0.05, enforce=True) as t:
        def work(i):
            try:
                with tokeymeter.bind_task(t):
                    return step(f"x-{i}")
            except tokeymeter.TaskEnvelopeExceeded:
                halted["n"] += 1
        with cf.ThreadPoolExecutor(8) as ex:
            list(ex.map(work, range(60)))
    assert halted["n"] > 0
    assert t.snapshot()["spend_usd"] < 0.20      # bounded, not 60 * 0.0155


def test_bind_task_rejects_a_non_state_object():
    with pytest.raises(TypeError):
        with tokeymeter.bind_task("not-a-task-state"):
            pass


def test_bind_task_accepts_none():
    with tokeymeter.bind_task(None):
        assert tokeymeter.current_task_id() is None


# ── validation ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("bad", ["", "has space", "has/slash",
                                 "http://x.com/y", "a" * 129, None, 123])
def test_bad_task_ids_rejected(bad):
    with pytest.raises((ValueError, TypeError)):
        with tokeymeter.task(bad):
            pass


@pytest.mark.parametrize("kw", [
    {"envelope": 0}, {"envelope": -1}, {"envelope": float("nan")},
    {"envelope": float("inf")}, {"envelope": "not-a-number"},
    {"max_calls": 0}, {"max_calls": 2.5}, {"max_calls": float("nan")},
    {"max_repeats": -3}, {"envelope": True},
])
def test_bad_limits_rejected(kw):
    """NaN slips past `> 0` because every NaN comparison is False — which would
    silently disable the ceiling. Rejected explicitly. So is a bool, because
    True == 1 would pass silently as a $1 envelope."""
    with pytest.raises(ValueError):
        with tokeymeter.task("t", **kw):
            pass


@pytest.mark.parametrize("value", ["1.00", "0.5", 2, 1.5])
def test_numeric_strings_and_ints_accepted_for_envelope(value):
    """Deliberately forgiving where the value is unambiguous: a policy file
    supplies "1.00" as a string, and coercing it is correct. Ambiguous inputs
    (bool) and undecidable ones ("abc", NaN) are still rejected."""
    with tokeymeter.task("t", envelope=value) as st:
        assert st.snapshot()["envelope_usd"] == float(value)


# ── fail-open ───────────────────────────────────────────────────────────

def test_task_accounting_never_breaks_a_call():
    """Bookkeeping must never be the reason a request fails. Only a deliberate,
    configured breach raises."""
    step = _step()
    with tokeymeter.task("t") as st:
        st._fingerprints = None        # corrupt internal state on purpose
        step("a")                      # must still succeed
    assert len(_records()) == 1


def test_snapshot_outside_a_task_is_none():
    assert tokeymeter.task_snapshot() is None
    assert tokeymeter.current_task_id() is None


# ── the number this exists to produce ───────────────────────────────────

def test_cost_per_task_is_computable_from_the_ledger():
    step = _step(tag="support")
    for t in ("ticket-1", "ticket-2", "ticket-3"):
        with tokeymeter.task(t):
            for i in range(3):
                step(f"{t}-{i}")
    by_task = {}
    for r in _records():
        if not r["hit"]:
            by_task[r["task_id"]] = by_task.get(r["task_id"], 0.0) + r["estimated_cost"]
    assert set(by_task) == {"ticket-1", "ticket-2", "ticket-3"}
    assert all(math.isclose(v, 3 * 0.0155, abs_tol=1e-4) for v in by_task.values())


def test_declared_reserve_makes_the_envelope_a_hard_cap():
    """With an upper bound on any single call, the worst case is held BEFORE
    each call — so the call that would cross the line is never made and final
    spend NEVER exceeds the envelope."""
    step = _step()
    with tokeymeter.task("t", envelope=0.10, reserve=0.02, enforce=True) as st:
        with pytest.raises(tokeymeter.TaskEnvelopeExceeded):
            for i in range(50):
                step(f"d-{i}")
    snap = st.snapshot()
    assert snap["spend_usd"] <= 0.10          # hard cap, not a trigger
    assert snap["reserve_breaches"] == 0      # the bound held


def test_without_a_reserve_the_hold_adapts_after_the_first_call():
    """No declared bound: the task holds back the largest cost it has actually
    seen. Only the first call is unbounded; from then on the envelope is
    respected."""
    step = _step()
    with tokeymeter.task("t", envelope=0.10, enforce=True) as st:
        with pytest.raises(tokeymeter.TaskEnvelopeExceeded):
            for i in range(50):
                step(f"e-{i}")
    snap = st.snapshot()
    assert snap["max_observed_call_usd"] > 0
    assert snap["spend_usd"] <= 0.10 + snap["max_observed_call_usd"]


def test_under_declared_reserve_is_counted_not_hidden():
    """If a real call costs more than the declared bound, the guarantee no
    longer strictly holds — so the overrun is surfaced. A guarantee you cannot
    audit is not a guarantee."""
    step = _step()
    with tokeymeter.task("t", envelope=1.00, reserve=0.001, enforce=True) as st:
        try:
            for i in range(10):
                step(f"f-{i}")
        except tokeymeter.TaskEnvelopeExceeded:
            pass
    assert st.snapshot()["reserve_breaches"] > 0


def test_reserve_validated_like_every_other_limit():
    for bad in (0, -1, float("nan"), float("inf"), True, "abc"):
        with pytest.raises(ValueError):
            with tokeymeter.task("t", envelope=1.0, reserve=bad):
                pass


def test_max_calls_gives_a_cost_independent_ceiling():
    """The tighter guarantee when a single call could be expensive."""
    step = _step()
    with pytest.raises(tokeymeter.TaskCallLimitExceeded):
        with tokeymeter.task("t", envelope=1000.0, max_calls=3, enforce=True):
            for i in range(20):
                step(f"d-{i}")
    executed = [r for r in _records() if not r["hit"]]
    assert len(executed) == 3


# ── adversarial: the four defects found in the deep review ──────────────

def test_failing_calls_are_bounded():
    """A call that RAISES still consumed budget upstream. Counting only
    successful calls left an agent failing every call completely unbounded —
    the exact runaway this feature exists to stop."""
    @tokeymeter.cache(model="m")
    def flaky(p):
        set_reported_usage(3000, 800)
        raise ValueError("upstream 500")

    with pytest.raises(tokeymeter.TaskCallLimitExceeded):
        with tokeymeter.task("t", max_calls=5, enforce=True):
            for i in range(100):
                try:
                    flaky(f"d-{i}")
                except ValueError:
                    pass


def test_failing_loop_is_detected():
    @tokeymeter.cache(model="m")
    def flaky(p):
        set_reported_usage(3000, 800)
        raise ValueError("tool error")

    with pytest.raises(tokeymeter.TaskLoopDetected):
        with tokeymeter.task("t", max_repeats=4, enforce=True):
            for _ in range(50):
                try:
                    flaky("the same failing call")
                except ValueError:
                    pass


def test_inner_task_cannot_exceed_the_outer_envelope():
    """Spend rolls up the parent chain and the whole chain is checked, so a
    sub-agent can't bypass its parent's ceiling."""
    step = _step()
    with tokeymeter.task("outer", envelope=0.05, reserve=0.02,
                         enforce=True) as outer:
        with pytest.raises(tokeymeter.TaskLimitExceeded) as ei:
            with tokeymeter.task("inner", envelope=100.0, enforce=True):
                for i in range(50):
                    step(f"n-{i}")
    assert ei.value.task_id == "outer"          # the OUTER limit stopped it
    assert outer.snapshot()["spend_usd"] <= 0.05


def test_nested_spend_rolls_up_without_double_counting():
    step = _step()
    with tokeymeter.task("outer", envelope=100.0) as outer:
        with tokeymeter.task("inner", envelope=100.0) as inner:
            for i in range(4):
                step(f"d-{i}")
    assert inner.snapshot()["calls"] == 4
    assert outer.snapshot()["calls"] == 4        # counted once, not twice
    assert outer.snapshot()["spend_usd"] == pytest.approx(
        inner.snapshot()["spend_usd"])


def test_free_cache_hits_are_never_blocked_by_the_reserve():
    """A hit costs nothing upstream, so holding the reserve for one would
    refuse work that cannot possibly breach the envelope."""
    step = _step()
    with tokeymeter.task("t", envelope=0.02, reserve=0.02, enforce=True) as st:
        step("warm")                              # the one paid call
        for _ in range(10):
            step("warm")                          # all free hits — must serve
    snap = st.snapshot()
    assert snap["calls"] == 11
    assert snap["spend_usd"] == pytest.approx(0.0155, abs=1e-4)


def test_enforcement_survives_a_swallowed_halt():
    """Agent retry loops catch broadly. The ceiling is checked before each
    call, so swallowing it just means the next attempt is refused too — the
    failure mode is a busy loop, never a budget breach."""
    upstream = {"n": 0}

    @tokeymeter.cache(model="m")
    def step(p):
        upstream["n"] += 1
        set_reported_usage(3000, 800)
        return "r"

    with tokeymeter.task("t", envelope=0.05, reserve=0.02, enforce=True) as st:
        for i in range(200):
            try:
                step(f"d-{i}")
            except Exception:          # the broad handler every agent has
                pass
    assert upstream["n"] <= 3
    assert st.snapshot()["spend_usd"] <= 0.05


def test_fingerprint_map_is_capped_and_reports_eviction():
    """A long task with tens of thousands of DISTINCT calls must not grow the
    map without bound. Singletons are evicted first — they are never loop
    evidence."""
    tokeymeter.clear_registered_pricing()
    tokeymeter.register_pricing("m", input_per_1m=0.001, output_per_1m=0.001)

    @tokeymeter.cache(model="m")
    def step(p):
        set_reported_usage(10, 5)
        return "r"

    with tokeymeter.task("long", envelope=1e9) as st:
        for i in range(12_000):
            step(f"unique-{i}")
    snap = st.snapshot()
    assert snap["distinct_fingerprints"] <= 10_000
    assert snap["fingerprints_evicted"] > 0


def test_loop_still_detected_after_eviction_pressure():
    """Eviction must not blind loop detection: a repeating fingerprint is kept
    because only count==1 entries are dropped."""
    tokeymeter.clear_registered_pricing()
    tokeymeter.register_pricing("m", input_per_1m=0.001, output_per_1m=0.001)

    @tokeymeter.cache(model="m")
    def step(p):
        set_reported_usage(10, 5)
        return "r"

    with pytest.raises(tokeymeter.TaskLoopDetected):
        with tokeymeter.task("t", max_repeats=3, enforce=True):
            for i in range(11_000):
                step(f"filler-{i}")
            for _ in range(10):
                step("the repeated one")


# ── integration: the number this feature exists to produce ──────────────

def test_chargeback_groups_by_task_id():
    """Cost per resolved task must be reachable through the SHIPPED report,
    not only by hand-rolling over raw records."""
    step = _step(tag="support")
    for t in ("ticket-1", "ticket-2", "ticket-3"):
        with tokeymeter.task(t):
            for i in range(3):
                step(f"{t}-{i}")
    cb = tokeymeter.chargeback_report(group_by=("task_id",))
    assert {r["task_id"] for r in cb["rows"]} == {"ticket-1", "ticket-2", "ticket-3"}
    assert all(r["executed_requests"] == 3 for r in cb["rows"])
    # reconciliation by construction still holds with the new dimension
    assert abs(sum(r["spend_usd"] for r in cb["rows"])
               - cb["totals"]["spend_usd"]) < 1e-9


def test_task_id_composes_with_other_dimensions():
    step = _step(tag="support")
    with tokeymeter.task("tk-1"):
        step("a")
    cb = tokeymeter.chargeback_report(group_by=("tag", "task_id"))
    assert cb["rows"][0]["tag"] == "support"
    assert cb["rows"][0]["task_id"] == "tk-1"


def test_task_composes_with_every_identity_field():
    from tokeymeter.engines.execution.endpoint import endpoint
    tokeymeter.register_key("prod", "sk-test-x")
    step = _step(tag="support")
    with tokeymeter.task("tk-1"), tokeymeter.principal("maya"), \
            endpoint("vllm-a100"), tokeymeter.key("prod"):
        step("a")
    r = _records()[0]
    assert r["task_id"] == "tk-1"
    assert r["principal"] == "maya"
    assert r["endpoint_identity"] == "vllm-a100"
    assert r["key_name"] == "prod"
    assert r["tag"] == "support"


def test_shadow_records_carry_the_task_id():
    @tokeymeter.cache(model="m", shadow=True)
    def step(p):
        set_reported_usage(3000, 800)
        return "r"
    with tokeymeter.task("shadow-t"):
        step("x")
        step("x")
    assert all(r["task_id"] == "shadow-t" for r in _records())


def test_chargeback_groups_by_agent():
    """task_id answers 'what did this ticket cost'; agent answers 'what does
    the diagnostic agent cost us' — the aggregate finance budgets against."""
    step = _step(tag="cat-digital")
    for i in range(6):
        with tokeymeter.task(f"diag-{i}", agent="service-diagnostic"):
            step(f"d{i}")
    for i in range(4):
        with tokeymeter.task(f"warr-{i}", agent="warranty-triage"):
            step(f"w{i}")
    cb = tokeymeter.chargeback_report(group_by=("agent",))
    assert {r["agent"] for r in cb["rows"]} == {"service-diagnostic",
                                                "warranty-triage"}
    assert abs(sum(r["spend_usd"] for r in cb["rows"])
               - cb["totals"]["spend_usd"]) < 1e-9


def test_agent_maps_to_a_gl_account():
    """Cost per agent has to reach the general ledger, or finance can see it
    and still not book it."""
    step = _step(tag="cat-digital")
    with tokeymeter.task("diag-1", agent="service-diagnostic"):
        step("a")
    cb = tokeymeter.chargeback_report(group_by=("agent",))
    gl = tokeymeter.general_ledger_rows(
        chargeback=cb, account_mapping={"service-diagnostic": "6100-AI-SVC"})
    assert gl["rows"][0]["account"] == "6100-AI-SVC"
