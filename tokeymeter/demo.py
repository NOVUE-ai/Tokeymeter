"""`tokeymeter demo` -- the self-hosted story in one command. Offline, no keys.

Three acts on the real engine (no mocks, in-memory ledger, leaves no files):

  ACT 1  An unpriced OSS model produces USD the report itself FLAGS as
         resting on the generic fallback -- nothing fabricated passes silently.
  ACT 2  Two measured inputs (GPU-hour cost, cluster throughput) derive the
         model's TRUE per-token rate, derivation shown, verifiable by hand.
  ACT 3  The same workload now reports honest USD -- and GPU-HOURS RECLAIMED,
         the savings unit a self-hoster actually disputes least.

Act 4 (allocated-vs-observed GPU: utilization, idle cost, GPUs reclaimable)
lives in the TokeNet control plane; the demo points there.
"""
from __future__ import annotations

import json

# Illustrative inputs -- the demo says so out loud. Real deployments measure
# their own: gpu_hour_rate = amortized cluster $/GPU-hour; throughput from
# the serving stack's metrics (e.g. vLLM /metrics).
GPU_HOUR_RATE = 2.10
THROUGHPUT_TPS = 1400
MODEL = "demo-oss-model"


def _rule(title: str) -> None:
    print("\n" + "=" * 66 + f"\n{title}\n" + "=" * 66)


def run() -> int:
    import tokeymeter
    from tokeymeter.storage import MemoryStore

    tokeymeter.set_default_store(MemoryStore())
    tokeymeter.set_in_memory_savings(True)
    tokeymeter.reset_savings()
    tokeymeter.clear_registered_pricing()

    print("tokeymeter demo -- self-hosted walkthrough (offline, no keys, "
          "no files written)")
    print(f"illustrative inputs: gpu_hour_rate=${GPU_HOUR_RATE}/hr, "
          f"measured_throughput={THROUGHPUT_TPS} tok/s -- substitute your own.")

    @tokeymeter.cache(model=MODEL)
    def ask(prompt: str) -> str:
        return "answer " * 120

    _rule("ACT 1 -- unpriced model: the report refuses to launder the fallback")
    for _ in range(3):
        ask("summarize the daily risk report " * 10)
    rep = tokeymeter.savings_report()
    print(json.dumps(rep["pricing"], indent=2))
    ok1 = rep["pricing"]["all_priced"] is False
    print("-> USD above rests on the generic fallback, and the report SAYS SO."
          if ok1 else "!! expected the fallback to be flagged")

    _rule("ACT 2 -- derive the TRUE rate from two measured numbers")
    d = tokeymeter.register_selfhost_pricing(
        MODEL, gpu_hour_rate_usd=GPU_HOUR_RATE,
        measured_tokens_per_second=THROUGHPUT_TPS)
    print(json.dumps(d, indent=2))
    print(f"-> ${d['usd_per_1m_tokens']:.4f}/1M tokens from YOUR cluster's "
          "economics -- derivation shown, checkable by hand.")

    _rule("ACT 3 -- same workload: honest USD + GPU-hours reclaimed")
    tokeymeter.reset_savings()
    for _ in range(50):
        ask("summarize the daily risk report " * 10)  # 1 miss, 49 hits
    rep = tokeymeter.savings_report()
    print(f"calls={rep['total_calls']}  hits={rep['cache_hits']}  "
          f"hit_rate={rep['hit_rate_pct']}%  all_priced="
          f"{rep['pricing']['all_priced']}  saved_usd=${rep['estimated_saved_usd']}")
    cap = tokeymeter.capacity_report(
        measured_tokens_per_second=THROUGHPUT_TPS,
        gpu_hour_rate_usd=GPU_HOUR_RATE)
    print(json.dumps(cap, indent=2))
    ok3 = rep["pricing"]["all_priced"] is True and cap["gpu_seconds_reclaimed"] > 0
    print("-> every dollar traces to a registered, derived rate; the "
          "GPU-seconds figure is capacity returned to your cluster."
          if ok3 else "!! act 3 invariants not met")

    _rule("ACT 4 -- allocated vs observed (utilization, idle cost, GPUs needed)")
    print("Lives in the TokeNet control plane (self-host reconciliation).\n"
          "In the repo: PYTHONPATH=. python scripts/selfhost_walkthrough.py")

    passed = ok1 and ok3
    print("\n" + ("ALL ACTS PASSED -- nothing assumed, nothing fabricated."
                  if passed else "DEMO FAILED -- see !! lines above."))
    _rule("KERNEL PATH (W3): the one-line Runtime, offline")
    from tokeymeter.runtime.facade import Runtime as _RT
    _rt = _RT(call=lambda p: "demo-model says: " + p[:40], receipt="never")
    print("Runtime.execute ->", _rt.execute("show me the kernel pipeline"))
    print("engine trace   ->", " -> ".join(
        t["engine"] + "." + t["phase"] for t in _rt.last.trace))
    ok, _ = _rt._trust.verify()
    print("trust chain    ->", "verified OK" if ok else "BROKEN")

    return 0 if passed else 1
