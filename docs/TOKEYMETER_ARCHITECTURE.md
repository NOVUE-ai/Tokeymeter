# Tokeymeter — Architecture Spec (v0.1)

**A NOVUE product · the token-economics layer**

> Not a gateway. Not a proxy. An **in-process, local-first SDK** that cuts the tokens your LLM calls actually pay for — deeper than any gateway bothers to — and shows you exactly how much, per workload, with no inflated claims.

---

## 0. Positioning (what makes it different, in one breath)

The gateway field (LiteLLM, Portkey, Bifrost, Helicone, Kong) is mature and crowded. They are **proxies**: your request hops to their server, gets routed/cached/tracked, then goes to the model. They compete on routing, failover, and basic semantic caching.

Tokeymeter does **not** fight that fight. It is a different shape:

| Them (gateways/proxies) | Tokeymeter (NOVUE) |
|---|---|
| A network hop / proxy server | **In-process** — runs inside your app, zero extra hop |
| Caching + routing as headline | **Deep token reduction** (compression) as headline |
| Data flows through their server | **Local-first** — nothing leaves except the real model call |
| Replace your stack | **Gateway-agnostic** — keep LiteLLM/Portkey, add Tokeymeter underneath |
| Provider-specific integrations | **Works with any API on earth** — wrap any callable |
| "Trust us" | **Provable** (the audit/proof sibling product) |

**The one-liner:** *You already have a gateway. Tokeymeter cuts the tokens it's still paying for — in-process, local, measured honestly — and plugs into anything.*

The four ownable claims:
1. **In-process, not a proxy** — no hop, microsecond overhead, no server to run.
2. **Local-first / trusted** — content stays in your process; we see patterns, never secrets.
3. **Gateway- and provider-agnostic** — wraps *any* callable that takes text and returns text. Any API in the world, including ones that don't exist yet.
4. **Deep compression** — the token-reduction seam the proxies ignore.

---

## 1. Design principles (non-negotiable)

- **In-process first.** The primary form is a library you `pip install` and wrap a function with. No mandatory server, no proxy, no hop.
- **Honest measurement.** Every saving is *measured* (real token deltas × real list price), never asserted. Claims are per-workload ranges, never a flat "60–80%."
- **Lossless where promised, bounded where not.** Cache hits and single-flight are exact. Compression carries an explicit quality budget and a fallback.
- **Gateway-agnostic.** Tokeymeter sits *below* or *beside* any gateway. It never demands replacing one.
- **Zero required dependencies in the core.** Stdlib-only hot path; heavy methods (e.g. learned compressors) are opt-in extras.
- **Fail-open.** If any optimizer errors, the original call still succeeds; the degradation is logged.
- **Content-blind learning.** Improvement comes from aggregated patterns / opt-in metadata, never customer content.

---

## 2. The pipeline (what happens to a call)

A wrapped call flows through ordered, independently-toggleable stages. Each stage either reduces cost or proves a saving; none may break the call.

```
your_fn(prompt) ──► [ Tokeymeter ] ──► real model call (only if needed)
                         │
   1. Normalize  ─ canonicalize prompt for matching
   2. Single-flight ─ collapse concurrent identical calls
   3. Exact cache ─ return stored answer (lossless)
   4. Semantic cache ─ return near-equivalent answer (bounded)
   5. Compress ─ shrink the prompt/context sent to the model
   6. Route ─ pick cheapest model meeting the quality bar
   7. (call model)
   8. Meter ─ record tokens, cost, latency, savings
```

Stages 1–4 avoid the call entirely (max saving). Stage 5 shrinks what's billed. Stage 6 picks cheaper compute. Stage 8 is the honesty layer.

---

## 3. The differentiator: deep compression (stage 5)

This is where Tokeymeter is unique. Gateways cache; they do not compress the tokens themselves. Grounded in real research:

### 3.1 Prompt / context compression
- **Perplexity-based token pruning** — the LLMLingua family (Microsoft Research): a small model scores token informativeness; low-information tokens are dropped. Reported up to ~20× compression at ~1.5% quality loss. *Status: integrate (opt-in extra; needs a small scorer model).*
- **Extractive / selective context** — drop redundant context spans by salience. *Status: buildable now, stdlib-friendly with heuristics; better with a scorer.*
- **Semantic dedup** — remove repeated/near-repeated context across a prompt. *Status: buildable now.*

### 3.2 Multi-turn / memory compression
- **Conversation summarization + delta encoding** — replace full history with a compact running state + the new turn. *Status: partially built (memory module).*
- **Reference-by-pointer context** — store long context once, send a handle + the delta. *Status: near-term.*

### 3.3 The honesty rule for compression
Every compression carries: **compression ratio, quality budget, and fallback**. If the budget can't be met, Tokeymeter sends the original. The audit records what was removed and why. **No silent quality loss.**

---

## 4. The other stages (parity done well, not the differentiator)

- **Exact cache** — hash-normalized prompt → stored response. Lossless. *Built.*
- **Semantic cache** — embedding similarity over a threshold returns a prior answer. Bounded; high-stakes never cached. *Built.*
- **Single-flight** — concurrent identical calls collapse to one. Lossless. *Built.*
- **Routing / cascades** — cheap model first, escalate on low confidence (FrugalGPT/RouteLLM line). *Near-term; the research is proven.*
- **Metering** — every call logged: model, tokens in/out, est. cost, latency, hit type, compression ratio, savings. The honesty engine. *Built (savings.py).*

---

## 5. The integration surface (ease-of-use is a feature)

The whole pitch dies if it's not trivial to adopt. Target:

```python
from tokeymeter import meter

# wrap ANY callable — any provider, any API on earth
@meter(model="gpt-4o-mini", tag="support")
def ask(prompt): return openai_call(prompt)   # or anthropic, gemini, local, custom

# or wrap inline
answer = meter(ask)("...")
```

- **One decorator / one wrap.** No config required to start.
- **Works with any callable** — OpenAI, Anthropic, Gemini, local llama.cpp, a REST call to a model that ships next year. Tokeymeter only assumes "text in, text out."
- **Gateway-friendly** — wrap your LiteLLM/Portkey call; Tokeymeter optimizes underneath it.
- **`tokeymeter report`** — one command prints measured savings, per tag/model/workload.

---

## 6. What's built vs. near-term vs. research-frontier (honest)

| Capability | Status |
|---|---|
| Exact cache, semantic cache, single-flight | **Built** |
| Metering / savings ledger | **Built** |
| Basic compression (heuristic, dedup) | **Built (extend)** |
| One-wrap decorator, any-callable surface | **Mostly built (harden)** |
| LLMLingua-class perplexity compression | **Near-term (opt-in extra)** |
| Routing / cascades | **Near-term** |
| Multi-turn delta/pointer compression | **Near-term** |
| Learned task-aware compression (>1000× regimes) | **Research-frontier** |

---

## 7. NOVUE alignment

- **This is the wedge, not the empire.** Tokeymeter is the felt-now cost product. The audit/proof/governance sibling is the durable moat. Same in-process core; shipped second.
- **Local-first thesis, embodied.** Being in-process (not a proxy) *is* the NOVUE difference made concrete at the smallest scale.
- **Content-blind leverage.** Compression/routing/cache models improve from aggregated patterns and opt-in signals — never customer content — consistent with the moat.
- **The beginning of something larger.** The same pipeline that meters tokens today becomes the plane that governs and proves actions tomorrow. We start at cost because it's the sharpest wedge; the architecture is built to grow.

---

## 8. Build order (next steps)

1. **Harden the core** — cache + single-flight + metering rock-solid, the one-wrap surface clean.
2. **Make compression the star** — extend heuristic compression now; wire an opt-in LLMLingua-class extra.
3. **Honest benchmark harness** — measure *stacked* savings per workload (cache-friendly, compression-heavy, multi-turn, cache-hostile). Publish ranges, not a flat number.
4. **Synthetic stress test** — all aspects, concurrency, failure injection, fail-open verification.
5. **Real-environment test** — against live providers, real workloads, every scenario.
6. **The README** — the one-wrap promise + measured numbers + 5-min install.
