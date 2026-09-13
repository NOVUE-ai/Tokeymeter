% TOKEYMETER — Full Architecture, Wedge & Honest Assessment
% NOVUE's prime product · the token-economics layer
% v1.0 · Internal / confidential

---

## 0. One sentence

Tokeymeter is an in-process, local-first Python library that **lowers your LLM
bill — deeper than any gateway bothers to — and proves how much, per workload,
without ever seeing your data leave your process.**

---

## 1. The wedge (why we are not "another gateway")

The LLM gateway field is crowded and mature: LiteLLM, Portkey, Bifrost, Kong,
Helicone. They are **proxies** — your request hops to their server, gets routed,
cached, and tracked, then goes to the model. They compete on routing, failover,
and basic semantic caching, and on raw proxy speed (Bifrost: ~11µs at 5000 RPS).

Tokeymeter does not fight on that axis. It is a different shape on four axes that
the proxy model structurally cannot match:

| Axis | Gateways / proxies | Tokeymeter |
|---|---|---|
| **Form** | A network hop / server you run | **In-process** — runs inside your app, no hop, no server |
| **Cost approach** | Caching + routing as headline; *observe* cost | **Deep token reduction** + the full stack; *reduce* cost at call time |
| **Data** | Request flows through their infra | **Local-first** — nothing leaves except the real model call |
| **Coupling** | Replace your stack | **Gateway- & provider-agnostic** — wraps any callable; keep your gateway |

The one-line wedge: **"You already have a gateway. Tokeymeter cuts the tokens it's
still paying for — in-process, local, measured honestly — and plugs into anything."**

And against the observability tools (Langfuse, Helicone, Datadog) that *chart*
cost by ingesting prompts: **"Everyone charts your LLM costs. Tokeymeter lowers
them — automatically, in-process, provably — then shows you the smaller bill."**

---

## 2. The full architecture — an ordered pipeline

A wrapped call flows through ordered, independently-toggleable stages. It is a
**single pass, cheapest-wins-first** — not a permutation search (which would burn
the savings it seeks by making multiple model calls).

```
  call
   │
   ▼
  1. normalize          canonical form for matching
  2. single-flight      collapse concurrent identical calls      ┐ lossless:
  3. exact cache        return stored answer if seen             │ avoid the
  4. semantic cache     return near-equal answer over threshold  ┘ model call
   │   (miss → continue)
   ▼
  5. route / cascade    easy→cheap model, hard→capable (conservative)
  6. compress (gated)   shrink the prompt — 4 quality safeguards
   │
   ▼
  7. MODEL CALL         whichever tier was chosen
   │
   ▼
  8. meter / record     tokens, cost, savings, fidelity — per model+tag
```

**Stages 1–4 (lossless)** can avoid the model call entirely — the largest possible
saving, and zero quality risk because the prompt is never altered.
**Stage 5 (route/cascade)** picks cheaper compute when it's safe to.
**Stage 6 (compress)** reduces the billed tokens, behind four safeguards (§4).
**Stage 8 (meter)** is the honesty layer — every figure measured, never asserted.

### 2.1 The Camp A method set (all built)

| Method | What it does | Status | Research base |
|---|---|---|---|
| Exact cache | Identical prompt → stored answer (lossless) | ✅ | Standard |
| Semantic cache | Near-equivalent prompt → cached answer (bounded) | ✅ | GPTCache line |
| Single-flight | Concurrent identical calls collapse to one | ✅ | Request coalescing |
| Memory reduction | Summarize/reuse history instead of resending | ✅ | Context-management practice |
| **Salience compression** | Model-free segment pruning (self-info + structure + dedup) | ✅ | Selective Context (Li et al.) |
| Safety gate + fidelity breaker | 4-layer quality protection on compression | ✅ | Circuit-breaker pattern |
| Model routing | Easy→cheap, hard→capable, conservative | ✅ | RouteLLM, FrugalGPT |
| Cascade | Cheap-first, escalate on low confidence | ✅ | FrugalGPT, cascade-routing (Dekoninck 2025) |
| LLMLingua (deep) | Perplexity-based compression | ✅ opt-in extra | LLMLingua-2 (Microsoft) |

### 2.2 Camp B — deliberately future, not out of scope

Quantization, distillation, pruning, batching, KV-cache management, PagedAttention,
spot instances — these reduce cost **only if you host and serve the model yourself.**
Tokeymeter sits in front of an API you *call*; for API users the provider already did
all of Camp B. So Camp B is **future scope for a NOVUE serving layer**, connected to
Camp A on the NOVUE platform later — never claimed as something the in-process library
does today. Labeled clearly so it never becomes a false present-tense claim.

---

## 3. Purpose & utility — for every part of the user base

The same narrow product reaches everyone *because the pain (token cost) is universal*
and the architecture (local, in-process) unlocks the high-trust segments. Reach comes
from positioning + architecture, not from widening the product.

| User | First pain | What Tokeymeter does | Why they stay |
|---|---|---|---|
| **Indie / hobby dev** | "My side-project LLM bill is bleeding." | One decorator → instant cache + compression savings; `report()` shows it. | Free, local, zero-config, saves real money. |
| **Developers** | Messy cost, fear of magic breaking output. | Drop-in savings + 4 quality safeguards + shadow mode to verify safely. | Trust: it never silently breaks output; measured. |
| **Startups** | Shipping fast, costs scaling faster. | Cache+route+compress+cascade stacked; per-tag attribution. | Cost control without re-architecting; scales. |
| **MSMEs** | Want AI, can't centralize/professionalize data. | Runs locally over their own calls; no data leaves; simple. | Affordable, private, no infra team needed. |
| **Consulting / finance** | High-consequence, audit, data leakage. | Local-first + high-stakes never-cache + provable savings. | Data never leaves; defensible to compliance. |
| **Healthcare / life-sci** | Protected data, can't send prompts out. | In-process compression/routing; content never egresses. | Legally adoptable where proxies are not. |
| **Aerospace / defense** | Air-gapped, sovereign, zero-egress. | Pure-stdlib core runs offline/air-gapped; no network. | The only shape they're *permitted* to adopt. |
| **AI labs / tech cos** | Massive token spend, want control. | Deep compression + routing + cascade + opt-in learned tiers. | Depth + extensibility (custom scorers). |
| **Large enterprises** | Model sprawl, cost opacity, audit burden. | Fleet view (via TokeyVue), policy, proof, attribution. | Becomes the standard cost-control layer. |

The cross-cutting truth: **lossless levers (cache, single-flight) serve everyone with
zero risk; the local-first architecture is what makes the regulated, high-value
segments reachable at all.**

---

## 4. Quality — the discipline, stated honestly

"Quality must not be compromised" is the right instinct; the honest version is: **lossless
levers are truly lossless; lossy levers are gated, measured, capped, and auto-disabled
when quality drops.** There are four independent safeguards on compression:

1. **Query-aware protection** — when the question is known, answer-bearing context is
   protected from pruning. (Measured: 5/5 facts preserved vs 2/5 blind.)
2. **SafeCompressor gate** — ships the compressed prompt only if self-verification passes
   (ratio band, query-term survival); else falls back to the original. Rare, counted.
3. **Conservative reduction cap** — refuses compressions that remove too much.
4. **Measured-fidelity circuit breaker** — per-workload; if measured compressed-vs-original
   output fidelity drops, it auto-stops compressing that workload until it recovers.

Routing/cascade add their own discipline: **conservative — uncertain prompts route up**;
cascades escalate on low-confidence answers. High-stakes calls bypass all optimization.

**The honest claim** (the only one that survives a technical buyer): *"Quality is protected
by four independent safeguards and continuously measured, with automatic fallback"* —
**not** *"zero risk."* The safeguards exist in code; what remains is proving they hold on
**real** traffic (the fidelity breaker needs real model outputs to populate). That is exactly
what the enterprise stress-test and your real-environment validation are for.

---

## 5. Is the architecture unique, novel, reliable, efficient? (Honest)

### 5.1 Unique — **Yes, on a real axis.**
No major competitor offers an **in-process, local-first, model-free deep-compression**
cost layer that is gateway-agnostic. The gateways are proxies; the observability tools
ingest content; the compression research (LLMLingua etc.) needs a model. The combination
— *in-process + content-blind + model-free deep compression + honest measurement + the
four-layer safety system* — is not offered by any single competitor today. That is a
genuine, defensible position, not a marketing gloss.

### 5.2 Novel — **Partly. Be precise about what.**
- **Not novel:** the individual methods. Caching, routing, cascades, compression are all
  published, proven techniques. We use them honestly as prior art — and we should *say*
  so; claiming we invented them would be false and fragile.
- **Genuinely novel (our contribution):** (a) the **model-free SalienceCompressor** —
  doing deep, query-aware compression with *zero dependencies* via corpus-free
  self-information, where the literature reaches for a model; (b) the **four-layer
  compression safety system** combining query-aware protection + confidence gate +
  reduction cap + measured-fidelity circuit breaker; (c) the **packaging** — an in-process,
  content-blind, gateway-agnostic library that unifies the whole Camp A stack behind one
  honest-measurement surface. The novelty is in the *synthesis and the safety discipline*,
  not the primitives. That is an honest and still-strong claim.

### 5.3 Reliable — **Architecturally yes; empirically: synthetic yes, real pending.**
- **Strong:** 484 passing tests; fail-open everywhere (any optimizer error → original call
  still succeeds); never-raises compression; falls-open safety gate; frozen public-API
  snapshot guarding against accidental breakage.
- **Honest gap:** all current numbers are synthetic-but-realistic. Reliability under real
  concurrency, adversarial inputs, and real model outputs is **measured next** (enterprise
  stress test) and **validated by you** (real-provider run). Until then: "reliable by
  construction and by synthetic test; real-traffic validation in progress."

### 5.4 Efficient — **Yes for the in-process design; with one honest caveat.**
- The hot path is stdlib-only; compression is O(n) over segments, microsecond-to-
  millisecond range; cache lookups are hash/embedding ops. No network, no model in the
  default path → negligible overhead vs the LLM call it's saving.
- **Caveat:** the *deep* tier (LLMLingua) is heavy (torch + model) and correctly opt-in;
  the *learned* router/cascade scorers (future) add model cost. The default is efficient;
  the deep options trade efficiency for compression depth, by explicit choice.

### 5.5 The honest bottom line
The architecture is **uniquely positioned, selectively novel (synthesis + safety), reliable
by construction, and efficient by default.** Its differentiation is real and defensible.
Its biggest unproven claim is *real-world* quality and savings — which is precisely why the
next two steps (enterprise stress test, real-environment validation) matter more than any
further feature.

---

## 6. Where we can improvise / improve (honest roadmap)

- **Learned tiers (opt-in):** a small DistilBERT-class scorer for routing/cascade confidence
  and a learned compressor — the deep upgrades, kept opt-in to preserve local-first defaults.
- **Output-token control:** cap/structure outputs to cut completion tokens (a Camp A lever
  not yet built).
- **Cascade-routing unification (Dekoninck 2025):** combine prompt-based routing and
  response-based cascading into one decision for better cost-quality frontier.
- **Real tokenizer integration:** replace the token *estimator* with provider tokenizers for
  exact accounting (currently estimated).
- **Adaptive thresholds:** let routing/compression thresholds tune from measured fidelity per
  workload (the fidelity breaker already provides the signal).
- **Tier-2/3 platform hooks:** structural fingerprints + federated learning (the moat-safe
  data flywheel) — built on the NOVUE platform (TokeyVue), opt-in.
- **More provider price coverage** in the pricing table; auto-refresh.

---

## 7. Competition — how we hold the line toward NOVUE

- **Don't out-proxy the proxies.** We lose a speed war with Bifrost (Go, µs). We don't enter
  it — in-process means there's no proxy to be fast or slow.
- **Don't out-chart the observability tools.** We reduce first, then report. Charts are a
  feature, not the product.
- **Win on the axis only we hold:** in-process + content-blind + model-free deep compression
  + honest measurement + self-hostable. That axis leads directly into NOVUE: the same
  content-blind discipline becomes the platform (TokeyVue), the audit/proof product, and the
  governance layer. Tokeymeter is the wedge; NOVUE is the platform it earns.

---

## 8. Status & next steps

**Built & tested (484 passing):** full Camp A pipeline — cache, semantic cache, single-flight,
memory, salience compression (query-aware), 4-layer safety, routing, cascade, unified
`metered_route`/`cascade`, honest metering, the marketable `tokeymeter` API.

**Next, in order:**
1. **Enterprise-grade synthetic stress test & audit** — concurrency, adversarial inputs,
   failure injection, quality-preservation under load. Turns "safeguards exist" into "measured."
2. **Real-environment validation (you, with API keys)** — live providers, real workloads, the
   honest published numbers.
3. **README + 5-min install + runnable example** — the adoption unlock; the path to user #1.
4. **Problems → solutions doc**, then **TokeyVue** platform, then the rest of NOVUE.

**The one unchanging truth:** the library is good; it has zero users. Everything above is in
service of the first developer who installs Tokeymeter, saves real money, and tells the next.
