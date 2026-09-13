# Tokeymeter — Architecture & Working

*How the library works, stage by stage, and the specific problem each part solves.*
*Grounded in the v0.11 codebase as audited — measured behavior, not aspiration.*

---

## 0. What Tokeymeter is, in one breath

Tokeymeter is an **in-process, local-first Python library** that sits between your
code and any LLM API. It does two things, and the pairing is the point:

- **Cost** — it reduces the tokens you pay for, at call time (cache, compress,
  route, cascade, provider-cache optimization).
- **Proof** — it records every optimization decision into a tamper-evident,
  content-blind audit chain, so a saving is also a *provable* decision.

It is **not** a gateway/proxy (no network hop, no server to run) and **not** an
after-the-fact observability dashboard. It runs inside your process; nothing
leaves the machine except the model call you were already making.

**The problem it solves:** when an LLM bill grows, the cost is rarely the user's
question — it's the *context resent on every call* (system prompts, tool schemas,
RAG chunks, history). A 200-token request quietly becomes 10,000 tokens, times
every call. Existing tools either add a network hop (gateways) or just chart spend
after the fact (dashboards). Tokeymeter reduces the spend in-process **and** proves
what it did — which is the part that stops it from being a commodity cache.

---

## 1. Design stance (the invariants every component obeys)

| Principle | What it means | Why it matters |
|---|---|---|
| **In-process, not a proxy** | A decorator, not a network service. | No new hop, no infra, no new failure domain, no data egress. |
| **Local-first / content-blind** | The core makes no network calls; metrics and audit findings carry numbers and *masked* locators, never prompt content. | Regulated/security-sensitive buyers can govern AI without exposing prompts. |
| **Fail-open** | If any optimizer errors, the original call still runs. | A bug in Tokeymeter must never take down your app. (Genuine *model* errors still surface to you.) |
| **Lossless where promised, bounded where not** | Cache + single-flight never alter the prompt; compression/routing are guarded and measured. | Savings never silently degrade quality. |
| **Honest measurement** | Savings are real token deltas against real list prices; every lossy step's fallback is counted and reportable. | The numbers survive contact with a customer. |

---

## 2. The request lifecycle

One decorator, `@meter` (a.k.a. `cache`), wraps any text-in/text-out callable. A
request flows through an **ordered, single-pass, cheapest-wins-first** pipeline:
each stage that can satisfy the request without calling the model does so and
short-circuits. The orchestration lives in `decorator.py`.

```
@meter(...)  ─►  1 normalize/key
                 2 single-flight ──hit/collapse──► serve (no model call)
                 3 exact cache    ──hit──────────► serve (no model call)
                   ↳ semantic cache + memory reuse
                 4 route / cascade   (pick cheap vs capable)
                 5 compress (gated)  (shrink the dynamic part)
                 6 provider-cache opt (structure the static prefix)
                 7 MODEL CALL        (only if 2–3 missed)
                 8 meter + record    (always: tokens, cost, saving, proof)
```

Stages 1–3 are **lossless** (the answer is byte-identical to what the model would
return for that exact prompt). Stages 4–6 are **guarded** (measured, bounded,
fail-open). Stage 8 always runs — including on cache hits (records the saving) and
on fail-open paths (records the degradation).

---

## 3. Stage by stage — how it works and what it solves

### Stage 1 — Normalize & cache key
**How:** the prompt (and the parameters that affect the answer) are normalized and
hashed into a stable cache key.
**Solves:** gives every later stage a deterministic identity for "this request,"
so identical work can be recognized and collapsed.

### Stage 2 — Single-flight (concurrency collapse)
**How:** when many threads issue the *same* request at once, the first becomes the
**leader** and computes; the rest become **followers** that wait and receive the
leader's result. Coordination is **fully in-memory** — the leader publishes its
result into an in-flight holder and a bounded recent-results map, so followers and
late arrivers get the value from memory regardless of the backing store's read
latency. Works on the default persistent store with no Redis required; cross-machine
collapse is available via the Redis backend.
**Solves:** the **thundering herd**. A cold cache plus a traffic spike would
otherwise fire N identical expensive calls. *Measured: a 200-way simultaneous burst
collapses to **1** underlying model call.*
**Honest note:** in-process by design. Across separate processes/pods, use the
Redis backend for shared collapse.

### Stage 3 — Exact cache (+ semantic cache + memory reuse)
**How:** an identical prompt returns the stored response (lossless). Optionally,
*semantic* caching returns a near-duplicate's answer above a similarity threshold
(opt-in, bounded — not lossless). Conversation memory bounds multi-turn context to
a recent window instead of resending the whole history.
**Solves:** the single biggest real-world lever — **repeated work**. Support bots,
RAG over reused documents, and agent retries repeat prompts constantly.
**Honest note:** cache savings *equal your duplicate rate*. A repetitive support
workload saves a lot; an all-unique creative workload saves ~0. The library can't
invent your hit rate — it can only avoid paying for the repeats you actually have.

### Stage 4 — Routing & cascades
**How:** `metered_route` sends easy prompts to a cheap model and hard ones to a
capable model; a `cascade` tries the cheap model first and escalates only if the
answer looks low-confidence. The router is **conservative — when unsure, it routes
up** (to the capable model).
**Solves:** paying flagship prices for trivial prompts. Most traffic is easy; only
some needs the expensive model.
**Honest note:** cascades that escalate pay for *both* calls, so net savings depend
on the escalation rate — which `report()` measures.

### Stage 5 — Compression (gated, model-free)
**How:** the dynamic part of a prompt is shrunk before it's billed. Two tiers:
- **Structural** (default, zero-dependency): strips whitespace, politeness filler,
  duplicate few-shot examples.
- **Salience** (the differentiator): scores segments by self-information and
  *query relevance*, pruning low-value text while protecting answer-bearing
  segments. No model, no torch — pure stdlib.

Four safeguards apply in order: **query-aware protection** (segments relevant to
the question are never pruned) → **confidence gate** (a compressed prompt ships only
if it self-verifies, else the original is sent and the fallback is counted) →
**reduction cap** (refuses over-aggressive compressions) → **fidelity circuit
breaker** (per-workload, auto-disables compression if measured fidelity drops, until
it recovers).
**Solves:** **context bloat** — the RAG chunks and long system prompts that dominate
input cost.
**Honest note (important):** the *default* structural tier does almost nothing on
dense document prose (*measured ~0–0.1% on RAG context* — there's no filler to cut).
The **query-aware salience** tier is what compresses RAG: *measured ~31–48% token
reduction with the buried answer fact preserved* across 1.5K–20K-token contexts. The
"40%+ with facts lost" figure comes only from the unsafe *query-blind* mode. The
honest, safe claim is **~31% input reduction on bloated prompts while preserving
facts** — and near-zero on already-lean prompts.

### Stage 6 — Provider-cache optimization (Tier 1 + Tier A)
**How:** providers cache the *static prefix* of a prompt (OpenAI automatically;
Anthropic via `cache_control`) — but only if the prefix is genuinely stable.
`check_cache()` audits a structured prompt and reports what's breaking the cache;
`optimize_cache()` places a `cache_control` breakpoint at the optimal boundary
without reordering turns; `compress_tools()` trims verbose tool descriptions while
preserving names and schemas exactly. The **Tier A** detector finds volatile content
(timestamps, UUIDs, request/session IDs) sitting in the prefix and silently breaking
the provider cache.
**Solves:** the **silent cache-killer** — a timestamp in a system prompt that
defeats provider prefix caching on every request, costing 50–90% more than necessary.
**Honest note (verified):** Tier A **detects and recommends, never rewrites** — and
it is **content-blind**: the offending value is masked, never reproduced, so the
audit itself never leaks a secret. *Measured: all four volatile patterns detected,
no secret leaked into findings.*

### Stage 7 — Model call
**How:** the wrapped callable (your real provider call) runs — only if stages 2–3
didn't already satisfy the request — using the optimized prompt.
**Solves:** nothing on its own; it's your existing call. Everything around it is the
value.

### Stage 8 — Meter & record
**How:** every call records tokens, cost (real token deltas × list prices),
savings, compression ship/fallback rate, and latency — split by model and tag.
`report()` returns measured figures; **shadow mode** (`shadow=True`) measures what
you *would* have saved without changing behavior.
**Solves:** "is this actually working, and how much did it save?" — answered with
your own measured numbers, not a vendor's claim.
**Honest note:** token counts use a documented ~4-chars/token estimate (applied
identically to hits and misses, so ratios are fair), so absolute dollars are
approximate until real-tokenizer integration. Run `shadow=True` on your workload for
your real figure.

---

## 4. The proof spine (what makes a saving a *provable* decision)

Every stage emits a structured **decision record**; these stream into a
**hash-chained audit ledger**. The chain can be exported as a portable **proof** and
independently **verified**, and entries are **signed** (HMAC or Ed25519) so tampering
is detectable.

| Component | Role | Purpose it solves |
|---|---|---|
| Decision records | Structured "what was decided and why" per stage | Makes every optimization inspectable. |
| Audit ledger (hash chain) | Append-only, tamper-evident sequence | "Prove what happened" after an incident or dispute. |
| Proof export/verify | A slice of the chain an auditor can check offline | Evidence without giving the auditor your stack. |
| Signing (HMAC / Ed25519) | Cryptographic signatures + verification | Non-repudiation; tamper is provable, not trust-me. |

**Why it exists:** cost reduction alone is a commodity — anyone can cache. Attaching
**proof, provenance, and tamper-evidence** to each cost decision is the durable
differentiator and the moat. The audit chain is **per-process** by design (each
process owns its chain); cross-process aggregation belongs to the later platform
layer, not the wedge.

---

## 5. Security & local-first guarantees

| Guarantee | How it works | Purpose it solves |
|---|---|---|
| **Content-blind** | Core makes no network calls; metrics/findings carry numbers + masked locators only. | Prompts and PII never become telemetry or a leak vector. |
| **PII redaction** | A redactor scrubs sensitive content before it reaches the cache, the model path, or the audit chain. | Sensitive data doesn't get cached or logged. |
| **Security policy** | Opt-in runtime guards (`SecurityPolicy`) can *require* redaction, encrypted/keyed cache, or non-repudiable audit — enforced by the cache and audit paths. | Fail-safe production posture: turn a guard on and the library refuses to run unsafely. |
| **At-rest encryption** | Pluggable cipher backend encrypts cached values on disk. | Cache files aren't a plaintext leak. |
| **Integrity self-check** | The library can verify its own files against a signed manifest and scan its environment. | Governance infrastructure must itself be governable (supply-chain self-defense). |
| **Fail-open** | Every optimizer path degrades to "run the original call" on error. | Reliability: optimization can never be the reason your app breaks. |

---

## 6. What Tokeymeter deliberately is *not*

- **Not a gateway/proxy.** No routing infrastructure, failover server, or rate-limit
  proxy. If you run a gateway (LiteLLM, Portkey), Tokeymeter sits *underneath* it and
  cuts the tokens it's still paying for.
- **Not model-side optimization.** No quantization, batching, or KV-cache serving —
  those require hosting the model.
- **Not (yet) action governance or compliance packs.** Action allow/escalate/block
  (Omega) and SOC2/GDPR/HIPAA evidence packs are deliberately **carved out** to later
  NOVUE phases; the shipped library is the **cost + proof** wedge.

---

## 7. Honest limitations (read before trusting a number)

- **All savings/quality figures are synthetic.** They are real *measurements of
  constructed workloads*, validated by the test and benchmark harnesses — but they
  are **not validated against live provider traffic at scale**. Your numbers will
  differ; run `shadow=True` for your real figure. Live-key validation is the
  founder-run next step.
- **Cache savings = your duplicate rate.** Not a fixed percentage.
- **Safe compression ≈ 31%** on bloated/RAG prompts (query-aware, facts preserved);
  ~0% on lean prompts. The default structural tier is light by design.
- **Token counts are estimated**, so dollar figures are approximate until
  real-tokenizer integration.
- **Default cache is persistent and shared on disk** (`~/.tokeymeter/cache.db`),
  global to the user. Two processes on the same host with default settings share
  cached responses. For multi-tenant/multi-service isolation, use a per-namespace or
  per-process store. *(Isolation-by-default is an open design decision.)*

---

## 8. Measured reality (from running the harnesses this build)

| Capability | Measured (synthetic) | Reading |
|---|---|---|
| Cache (duplicate-heavy workload) | ~54% cost saved | = the workload's duplicate rate |
| Single-flight (200-way burst, default store) | 200 → **1** call | thundering-herd collapse works on the default store |
| Compression — query-aware salience (RAG) | ~31–48% tokens, fact preserved | the real, safe compression number |
| Compression — structural default (RAG) | ~0% | light tier; not for dense prose |
| Memory (multi-turn) | bounded context, ~17% | no quadratic history growth |
| Cache correctness (5,000 mixed reqs) | **0 mismatches** | never serves a wrong answer |
| Tier A volatile detection | 4/4 detected, content-blind | catches the silent cache-killer |

The machinery is correct and honest. The remaining step that turns these synthetic
figures into numbers you can put in front of a customer is **real-provider
validation on live keys** — which only you can run.
