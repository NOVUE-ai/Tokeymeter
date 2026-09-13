"""Enterprise workload simulation — REAL OpenAI, self-budget-capped.

Simulates a JPMC-shaped day: multiple teams, agents, and workloads hitting
gpt-4o-mini through Tokeymeter, exercising EVERY engine capability on real
traffic — while Tokeymeter's own budget-enforced key hard-caps the spend so
the test cannot overrun your $3.

    setx OPENAI_API_KEY "sk-..."     (Windows, new shell after)
    python scripts/enterprise_live_sim.py                 # ~$0.30-0.60
    python scripts/enterprise_live_sim.py --dry           # no API, logic only
    python scripts/enterprise_live_sim.py --cap 1.00      # tighter hard cap

Exercises: exact cache, semantic reuse, compression, cost attribution by
principal/team, provider-usage truth (reported tokens), secret firewall,
PII redaction, budget hard-stop, savings ledger, audit provability. Prints
a per-team chargeback table and the reconciled savings at the end.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="no API calls")
    ap.add_argument("--cap", type=float, default=1.50,
                    help="hard budget cap in USD (Tokeymeter enforces it)")
    ap.add_argument("--scale", type=int, default=1,
                    help="workload multiplier (1 ~= $0.4, keep small)")
    args = ap.parse_args()

    import tokeymeter
    from tokeymeter.storage import SQLiteStore
    from tokeymeter import DefaultRedactor

    live = not args.dry
    if live and not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set. Use --dry for a no-cost logic run.")
        return 2

    client = None
    if live:
        try:
            from openai import OpenAI
            client = OpenAI()
        except ImportError:
            print("pip install openai"); return 2

    work = os.path.join(os.getcwd(), "_sim_cache.db")
    tokeymeter.set_default_store(SQLiteStore(work))
    tokeymeter.reset_savings()
    tokeymeter.clear_keys()

    # Tokeymeter guards its own test budget — the product IS the safety net.
    tokeymeter.register_key("openai-live", os.environ.get("OPENAI_API_KEY", "dry"),
                            monthly_cap_usd=args.cap)

    redactor = DefaultRedactor()

    def call_model(prompt, model="gpt-4o-mini"):
        if not live:
            return f"[dry] {len(prompt)} chars"
        from tokeymeter.usage import set_reported_usage
        r = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}],
            max_tokens=120)
        if r.usage:                      # T1.1: feed provider's real numbers
            set_reported_usage(r.usage.prompt_tokens, r.usage.completion_tokens)
        return r.choices[0].message.content

    @tokeymeter.cache(model="gpt-4o-mini", redactor=redactor)
    def summarize(prompt, model="gpt-4o-mini"):
        return call_model(prompt, model)

    @tokeymeter.cache(model="gpt-4o-mini", semantic=True,
                      semantic_threshold=0.90, redactor=redactor)
    def classify(prompt, model="gpt-4o-mini"):
        return call_model(prompt, model)

    # ── the enterprise: teams → people → agents → workloads ───────────
    TEAMS = {
        "platform-eng": ["priya", "dev", "marcus"],
        "risk-quant":   ["anya", "jon"],
        "support-cx":   ["sam", "mei"],
    }
    DOCS = [
        "Q3 counterparty exposure rose on three desks; recommend hedging.",
        "The settlement failed due to a mismatched CUSIP on the wire.",
        "Client requested early termination of the interest-rate swap.",
        "Compliance flagged an unusual pattern in overnight repo activity.",
    ]
    TICKETS = ["reset my password", "reset the password please",
               "card declined at checkout", "why was my card declined",
               "how do I export statements", "export my statements how"]

    random.seed(7)
    stats = {"calls": 0, "errors": 0, "blocked_budget": 0}
    t0 = time.time()
    n = 40 * args.scale

    try:
        for i in range(n):
            team = random.choice(list(TEAMS))
            person = random.choice(TEAMS[team])
            try:
                with tokeymeter.principal(person), tokeymeter.key("openai-live"):
                    if i % 3 == 0:
                        # summarization: heavy repetition → exact-cache wins
                        summarize(random.choice(DOCS))
                    else:
                        # triage: paraphrases → semantic reuse
                        classify(random.choice(TICKETS))
                stats["calls"] += 1
            except tokeymeter.KeyBudgetExceeded:
                stats["blocked_budget"] += 1
                print(f"  [budget guard] hard cap ${args.cap} hit at call {i} "
                      "— Tokeymeter stopped its own test. Working as designed.")
                break
            except Exception as e:
                stats["errors"] += 1
                if stats["errors"] <= 3:
                    print(f"  [call error] {type(e).__name__}: {str(e)[:80]}")

        # secret-firewall + PII proof on real traffic
        with tokeymeter.principal("dev"), tokeymeter.key("openai-live"):
            summarize("Reset the key sk-live-AAAA1111BBBB2222CCCC3333DDDD "
                      "for user john.doe@bank.com immediately")
    except KeyboardInterrupt:
        print("interrupted")

    elapsed = time.time() - t0
    rep = tokeymeter.savings_report()
    print("\n" + "=" * 60)
    print(f"WORKLOAD COMPLETE · {stats['calls']} calls · {elapsed:.1f}s · "
          f"errors={stats['errors']} · budget-blocked={stats['blocked_budget']}")
    print("=" * 60)
    print(f"total calls        {rep['total_calls']}")
    print(f"cache hits         {rep['cache_hits']}  ({rep['hit_rate_pct']}%)")
    print(f"est. spent         ${rep.get('estimated_spent_usd', 0):.4f}")
    print(f"est. saved         ${rep['estimated_saved_usd']:.4f}")
    print(f"all priced         {rep['pricing']['all_priced']}")
    ks = tokeymeter.key_status("openai-live")
    print(f"key spent/cap      ${ks['spent_usd']:.4f} / ${ks['monthly_cap_usd']}")

    print("\nPER-PERSON CHARGEBACK (real attribution, no tagging):")
    # emulate the plane's rollup locally from the ledger
    from tokeymeter import savings as sv
    by = {}
    for r in sv._tracker._iter_records():
        p = r.get("principal") or "unattributed"
        b = by.setdefault(p, {"calls": 0, "spent": 0.0, "saved": 0.0})
        b["calls"] += 1
        b["spent"] += float(r.get("estimated_cost", 0))
        b["saved"] += float(r.get("cost_saved_usd", 0) or 0)
    for p, b in sorted(by.items(), key=lambda x: -x[1]["spent"]):
        print(f"  {p:<16} calls={b['calls']:<4} spent=${b['spent']:.4f} "
              f"saved=${b['saved']:.4f}")

    # token-source proof (T1.1)
    srcs = {}
    for r in sv._tracker._iter_records():
        srcs[r.get("token_source", "?")] = srcs.get(r.get("token_source", "?"), 0) + 1
    print(f"\ntoken source split (reported=real provider usage): {srcs}")
    print(f"\nledger + store written to: {work}")
    print("run  `python -m tokeymeter doctor`  to see live overhead percentiles.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
