# RouteLLM Architecture — Deep Analysis & Clean-Room Integration Design

*Senior-level engineering analysis. Extracted from the RouteLLM repo, docs, SDK
examples, and paper (Ong et al., ICLR 2025). Every architectural element mapped
to a clean-room implementation integrated with the Tokeymeter engine and sealed
into the NOVUE proof spine. No code copied — architecture and public math only.*

---

## Part 1 — RouteLLM's architecture, extracted precisely

### 1.1 The central abstraction (the one thing to take)
RouteLLM's entire design reduces to a single-method router interface:

> **`calculate_strong_win_rate(prompt) -> float`** — returns P(strong model
> meaningfully beats weak model | prompt), in [0,1]. If that win-rate **>**
> cost-threshold α, route to strong; else route to weak.

Every router — heuristic, BERT, matrix-factorization, causal-LLM — implements
*only* this method. The threshold comparison and the actual routing are shared
infrastructure. This is the cleanest idea in the codebase and the spine of our
integration: **one scoring method, swappable brains, shared decision logic.**

There is also a **`batch_calculate_win_rate(prompts) -> Series`** for offline
evaluation/calibration — same scorer, vectorized.

### 1.2 The four scoring brains (and their real tradeoffs)
| Router | How it scores win-rate | Weight | Our verdict |
|---|---|---|---|
| **sw_ranking** | Embed prompt; weighted Bradley-Terry (Elo) over similar labeled battles, weight ω=γ^(1+S) by similarity S | Carries the whole battle dataset | Training-free but heavy/slow; not for us |
| **mf** (their best) | Bilinear latent score s(model,q)=⟨model_emb, W·query_emb⟩; win = σ(s(strong)−s(weak)) | Tiny at inference (dim ~128) | **The idea to reimplement** — best cost/quality, lightweight |
| **bert** | Fine-tuned BERT classifier head → win prob | ~110M model | Simple, self-contained; a later opt-in brain |
| **causal_llm** | 8B instruction model predicts routing | Heaviest | Router cost rivals savings; skip |

### 1.3 Threshold calibration (we lack this — high-value to take)
RouteLLM calibrates the threshold to a *spend target* rather than hand-tuning:

> `calibrate_threshold --routers mf --strong-model-pct 0.5` → threshold = 0.11593
> (so ~50% of queries route to strong on the calibration distribution)

Mechanism (clean to reimplement): score a representative sample of prompts, take
the win-rate distribution, and set the threshold at the percentile that yields
the target strong-model fraction. **This decouples business policy ("spend ~50%
on the strong model") from the routing mechanism — pure logic, no data model.**

### 1.4 What RouteLLM does NOT do (the gaps we fill = the moat)
- **No caching** — it routes every call; repeated/duplicate calls all pay.
- **No proof** — no record of what was routed, why, or that a high-stakes call
  was *not* cheaped out. No audit, no content-blind ledger, no reconciliation.
- **No verify-then-escalate** — it *predicts and commits* (gambles); there is no
  post-response quality check that escalates a weak cheap answer.
- **No governance / accumulated state** — generic router, no per-customer tuning,
  nothing compounding.
- **Server mode is a network proxy** — their OpenAI-compatible server is the
  exact gateway pattern NOVUE is built against. (Library/Controller mode is fine.)

---

## Part 2 — The clean-room mapping to Tokeymeter

We take the **architecture and logic**, implement as our own code, integrate with
the engine, and seal into the moat. Five pieces:

### Piece 1 — Evolve our Router to the win-rate interface
Our current `Router.route()` scores *complexity*. We add a `score(prompt) ->
win_rate` method whose semantics match RouteLLM's: **"how likely is it that the
capable model is meaningfully needed here?"** This is a reframing of our existing
heuristic from "complexity" to "strong-model-win-probability" — the *same* cheap
signals (length, hard cues, code, multi-step) now interpreted as win-rate
evidence. Backward compatible: `route()` stays, now built on `score()`.

The win-rate framing is better than complexity because it optimizes the *right*
target (will cheap suffice?) — the John-at-Uber insight. A short hard prompt
("prove this lemma: …") scores high win-rate (needs strong) even though it's
short; a long easy prompt (summarize this doc) scores low.

### Piece 2 — Threshold calibration (new, pure logic)
Add `calibrate(prompts, target_strong_pct) -> threshold`: score the sample,
sort win-rates, pick the percentile cutting off the target fraction. Exactly
RouteLLM's calibration, our code, no dependency. Lets a customer say "route 15%
to the capable model" and get the threshold that achieves it on *their* traffic.

### Piece 3 — The single-method interface as the swap seam
Define an abstract `WinRateScorer` with one method `score(prompt)->float`. The
heuristic is the default implementation. A future trained scorer (our own MF from
public math, trained on a design partner's data) drops in behind the *same*
interface with zero wrapper changes. This is RouteLLM's `ROUTER_CLS` extensibility
pattern, our implementation — "swap the brain, keep the wiring."

### Piece 4 — Seamless engine integration (already wired, now upgraded)
Routing is already wired into the OpenAI wrapper (opt-in, before fingerprint,
fail-open). The evolved win-rate router slots into that same path. The cache key
still includes the routed model (so caching composes correctly on top of routing).
The Cascade (verify-then-escalate) remains the quality guarantee layered over it.

### Piece 5 — Seal routing verdicts into the proof spine (THE MOAT)
`DecisionRecord` already reserves `routed_model: Optional[str]` "for provable
routing (v0.11)" — the seam was designed. We complete it: every routing decision
records the chosen model, the win-rate score, the reason, and the estimated
saving, into the content-blind audit chain. **This is the thing RouteLLM
structurally cannot do** — provable cost governance: an auditor can verify that
no high-stakes call was silently routed cheap, and that the savings reconcile.
This is what makes it *NOVUE's* router, not a RouteLLM clone.

---

## Part 3 — Why this is the right design (not just a port)

**It optimizes the right thing.** Win-rate ("will cheap suffice?") not complexity
— the John-quality insight, taken from RouteLLM's core logic.

**It stays zero-dependency in the core.** The heuristic win-rate scorer + the
calibration + the cascade are pure Python — they go in the zero-dep core, air-gap
capable. The *trained* MF brain (when data exists) is the opt-in `[smart-routing]`
tier. Light core, opt-in power — the doctrine holds.

**It composes with the whole engine.** Routing feeds the cache (routed model in
the key), the cascade guarantees quality on top, and every decision is sealed in
the proof spine — none of which RouteLLM has. The router becomes one instrument in
the control plane, not a standalone tool.

**It threads the moat.** The routing verdict in the audit chain is provable cost
governance — defensible, un-copyable, compounding into TokeNet's accumulated
state. RouteLLM routes; NOVUE *proves what it routed and why.*

**The trained brain has a clean path.** When a design partner's traffic exists, we
build our own matrix-factorization scorer from the public math (Koren 2009 latent
factor models; Bradley-Terry 1952; sigmoid-of-score-gap) — trained on *our/their*
data, behind the *same* `score()` interface. Never their code; their *idea*,
our implementation, our data. And it's tuned per-customer → accumulated moat.

---

## Part 4 — The honest boundary (what we are and aren't doing now)

**Doing now (Path A — buildable immaculately, no data, no dependency):**
- The win-rate `score()` interface (heuristic brain)
- Threshold calibration
- The swap seam (`WinRateScorer` abstraction)
- Sealing routing verdicts into the proof spine (completing the reserved field)
- Full integration with the existing wrapper + cascade

**Deferred (needs a design partner's data):**
- The trained matrix-factorization scorer (our own, from public math, on our data)
  — drops in behind the same interface later.

We are taking RouteLLM's **architecture** (single-method win-rate routing),
**logic** (route on quality-win not complexity), and **calibration mechanism**,
implementing them as **our own clean code**, integrating with the **engine**, and
sealing them into the **moat** — which is exactly "their logic, our engine, our
moat," done the legally clean way: from their published architecture and the
public math, not their source.
