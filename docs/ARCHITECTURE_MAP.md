# Tokeymeter — Architecture-to-Code Map

> **Purpose.** This document maps the architecture diagram onto the actual source
> tree, one stage at a time, and then shows how the stages integrate into a single
> pass. It is the reference a new engineer reads on day one to answer two
> questions: *"where does stage N live?"* and *"if I want to change X, which file
> do I open?"*
>
> Every module and symbol named here was verified against the source. Nothing is
> aspirational — this describes the code as it is at `0.11.0`.

---

## 1. One package, two roles

Everything ships as a single package, `tokeymeter`, with two cohesive
subpackages. There is no separate engine package — the implementation modules and
the curated product API live together under one name.

| Part | Path | Role |
|---|---|---|
| **Public product API** | `tokeymeter/__init__.py` (+ `tokeymeter/_api.py`) | The narrow, stable surface: `meter`, `report`, `reset`, `compressor`, `router`, `metered_route`, `cascade`, `optimize_cache`, `check_cache`, `compress_tools`, `no_optimize`. Curated wrappers, implemented in `_api.py` and re-exported. |
| **Implementation modules** | `tokeymeter/*.py` | The full pipeline: flat, single-responsibility modules (`decorator`, `storage`, `semantic`, `salience`, `router`, `cache_optimize`, `savings`, …). |
| ↳ Audit subpackage | `tokeymeter/audit/` | Hash-chained ledger, proofs, signers — the "proof" half of the cost+proof wedge. |
| ↳ Backends subpackage | `tokeymeter/backends/` | Pluggable cache backends: Redis store, at-rest cipher. |

**Design intent:** the public API is deliberately small and stable — most users
only ever touch `meter` and `report`. The richer implementation symbols
(`SalienceCompressor`, `Router`, `CacheOptimizer`, the audit/proof machinery)
are also importable for advanced use, but the curated wrappers in `_api.py` are
the front door. The wrappers never re-implement logic; they compose the
implementation modules. The package's `__version__` is the single source of truth
(set in `tokeymeter/__init__.py` and mirrored in `pyproject.toml`).

---

## 2. The request lifecycle (the seven stages)

The pipeline is **ordered, single-pass, cheapest-wins-first**: each stage that can
satisfy the request without calling the model does so and short-circuits. The
orchestration lives in `tokeymeter/decorator.py`, which is why that file is the largest
in the tree (1,735 lines — see §6 for the standing note on this).

```
@meter / metered_route / cascade            facade: tokeymeter/__init__.py
        │                                     orchestration: tokeymeter/decorator.py
        ▼
 1. normalize + single-flight  ── hit ──▶ return (no model call)
        ▼
 2. exact + semantic cache      ── hit ──▶ return (no model call)
        ▼
 3. route / cascade            (choose cheap vs capable)
        ▼
 4. compress (gated)           (reduce the dynamic part before billing)
        ▼
 5. cache optimization (T1+TierA)  (structure the static prefix; audit volatility)
        ▼
 7. MODEL CALL                 (only if stages 1–2 didn't satisfy it)
        ▼
 6. meter / record             (tokens, cost, savings, fidelity — always runs)
```

> Stage 6 (meter/record) is drawn after the model call because it records the
> outcome, but conceptually it wraps the whole pass: it runs on cache hits too
> (recording the saving) and on fail-open paths (recording the degradation).

---

## 3. Stage-by-stage map

### Stage 1 — Normalize + single-flight *(lossless)*

| Concern | Module | Key symbols |
|---|---|---|
| Cache-key normalization | `tokeymeter/utils.py` | `make_cache_key()` |
| In-process store + single-flight | `tokeymeter/storage.py` | `MemoryStore`, `SQLiteStore` |
| Orchestration | `tokeymeter/decorator.py` | single-flight collapse path |

Identical concurrent calls collapse to one model call; the rest wait on the
leader's result. Lossless — the prompt is never altered. In-process by default
(no Redis required).

### Stage 2 — Exact + semantic cache + memory reuse *(lossless / bounded)*

| Concern | Module | Key symbols |
|---|---|---|
| Exact-match cache | `tokeymeter/storage.py` | `MemoryStore`, `SQLiteStore` |
| Semantic (near-duplicate) cache | `tokeymeter/semantic.py` | `SemanticCache`, `default_encoder()`, `is_available()`, `is_vec_index_available()` |
| Conversation/context reuse | `tokeymeter/memory.py` | `ConversationMemory`, `Turn`, `TruncationSummarizer`, `CallableSummarizer`, `InMemoryMemoryStore`, `SQLiteMemoryStore` |
| TTL/expiry envelope on stored values | `tokeymeter/envelope.py` | `wrap()`, `unwrap()`, `is_expired()` |
| Cache hit/miss events | `tokeymeter/events.py` | `CacheEvent`, `subscribe()`, `emit()` |
| Distributed backend | `tokeymeter/backends/redis_store.py` | `RedisStore` |
| At-rest encryption | `tokeymeter/backends/cipher.py` | `FernetCipher`, `NoOpCipher` |

Exact match is lossless. Semantic match is **bounded** (returns a prior answer for
a near-duplicate) — opt-in, only where acceptable.

### Stage 3 — Route / cascade

| Concern | Module | Key symbols |
|---|---|---|
| Easy→cheap, hard→capable routing | `tokeymeter/router.py` | `Router`, `RouteDecision` |
| Cheap-first, escalate-on-low-confidence | `tokeymeter/router.py` | `Cascade`, `CascadeResult` |
| Facade entry points | `tokeymeter/__init__.py` | `metered_route()`, `cascade()`, `router()` |

Conservative by design: when the router is unsure, it routes **up** to the capable
model, so saving money never silently degrades a hard answer.

### Stage 4 — Compress (gated) — *the differentiator*

| Concern | Module | Key symbols |
|---|---|---|
| Model-free salience compression | `tokeymeter/salience.py` | `SalienceCompressor` |
| Compressor protocol + structural/LLMLingua impls | `tokeymeter/compression.py` | `Compressor`, `StructuralCompressor`, `LLMLinguaCompressor`, `compose()`, `safe_compress()`, `CompressionResult` |
| Confidence gate + fallback stats | `tokeymeter/safe_compress.py` | `SafeCompressor`, `compression_stats()`, `reset_compression_stats()` |
| Query-aware span protection (structural IR) | `tokeymeter/prompt_ir.py` | `PromptIR`, `Span`, `parse()`, `reconstruct()`, `Permissions`, `permissions_for()` |
| Reduction cap + fidelity circuit breaker | `tokeymeter/decorator.py` | `_CompressionFidelityBreaker`, `set_compression_max_reduction()`, `set_fidelity_circuit_breaker()`, `compression_breaker_state()` |

**The four safeguards, and where each lives** (this is the heart of the quality
story):

1. **Query-aware protection** — `prompt_ir.py` (spans relevant to the query are
   marked protected and excluded from pruning).
2. **Confidence gate** — `safe_compress.py` (`SafeCompressor` ships the compressed
   prompt only if it self-verifies; otherwise the original is sent and the
   fallback is counted).
3. **Reduction cap** — `decorator.py` (`set_compression_max_reduction`).
4. **Fidelity circuit breaker** — `decorator.py` (`_CompressionFidelityBreaker`):
   per-workload, auto-disables compression if measured fidelity drops, until it
   recovers.

### Stage 5 — Cache optimization (Tier 1 + Tier A)

| Concern | Module | Key symbols |
|---|---|---|
| Prefix-cacheability + `cache_control` placement | `tokeymeter/cache_optimize.py` | `CacheOptimizer`, `CacheReport` |
| Tier A volatile-content detector (content-blind) | `tokeymeter/cache_optimize.py` | `VolatileFinding` |
| Tool-definition compression | `tokeymeter/cache_optimize.py` | `compress_tools()` |
| Facade entry points | `tokeymeter/__init__.py` | `optimize_cache()`, `check_cache()`, `compress_tools()` |

Tier A **detects and recommends, never rewrites**: it masks the offending value
(content-blind), names the finding (timestamp/UUID/ID/token breaking the provider
cache), and leaves the decision to the human. Stages 4 and 5 stack — compress the
*dynamic* part, cache the *static* prefix — two complementary cost layers.

### Stage 6 — Meter / record *(always runs)*

| Concern | Module | Key symbols |
|---|---|---|
| Per-call records + savings rollups | `tokeymeter/savings.py` | `SavingsTracker`, `CallRecord`, `savings_report()`, `reset_savings()` |
| Token + cost estimation | `tokeymeter/pricing.py` | `estimate_cost()`, `estimate_tokens()` |
| Prometheus export | `tokeymeter/metrics.py` | `PrometheusCollector` |
| Facade entry points | `tokeymeter/__init__.py` | `report()`, `reset()` |

Measures real token deltas against real list prices. `report(detail=True)` surfaces
the compression ship/fallback rate so you can confirm the compressor is actually
shipping, not silently falling back.

> **Honest caveat (carried from the README):** token counts are *estimated*, not
> provider-tokenizer-exact, so costs are approximate until real-tokenizer
> integration lands.

### Stage 7 — Model call

The wrapped callable — the customer's own provider call (OpenAI, Anthropic, local
model, any text-in/text-out function). The library reaches it only if stages 1–2
didn't satisfy the request. Related:

| Concern | Module | Key symbols |
|---|---|---|
| Cache pre-warming (offline) | `tokeymeter/warmup.py` | `warm_from_iterable()`, `warm_from_jsonl()` |

---

## 4. The cross-cutting spine

These modules are not a single pipeline stage; they are touched by many stages and
enforce the library's promises. This is the part that makes Tokeymeter more than a
cache.

| Concern | Module(s) | Key symbols | Promise it enforces |
|---|---|---|---|
| **Decision records** | `tokeymeter/decision.py` | `DecisionRecord`, `on_decision()` | Every stage emits a structured record of what it decided and why. |
| **Provable audit ledger** | `tokeymeter/audit/log.py` | `AuditLog`, `AuditEntry`, `Checkpoint`, `verify_entries()` | Hash-chained, per-process; "every action is a provable decision." |
| **Proofs** | `tokeymeter/audit/proof.py` | `export_proof()`, `verify_proof()`, `AuditProof` | A slice of the chain can be exported and independently verified. |
| **Signing** | `tokeymeter/audit/signers.py` | `HMACSigner`, `Ed25519Signer`, `load_or_create_install_secret()` | Entries are signed; tamper is detectable. |
| **Privacy / PII** | `tokeymeter/privacy.py` | `DefaultRedactor`, `redact()`, `Redactor` | PII never reaches the model, the cache, or the audit chain. |
| **Content-blind guarantee** | `cache_optimize.py`, `degraded.py`, `decision.py` | (masking throughout) | Metrics and findings carry numbers and masked locators — never prompt content. |
| **Fail-open observability** | `tokeymeter/degraded.py` | `emit_degraded()`, `on_degraded()`, `degraded_event_count()`, `DegradedEvent` | Any optimizer error → original call still succeeds; the degradation is *counted*, not silent. |
| **Action governance** | *(carved out)* | — | Allow/escalate/block for agent actions. Moved to the NOVUE governance seed (Omega, Phase 4); not part of the cost+proof wedge. |
| **Security policy** | `tokeymeter/policy.py` | `SecurityPolicy`, `set_security_policy()` | Runtime guardrails. |
| **Supply-chain self-defense** | `tokeymeter/integrity.py` | `self_check()`, `generate_manifest()`, `verify_self()`, `scan_environment()` | The library can verify its own files and scan for hijack vectors. |
| **Cache administration** | `tokeymeter/admin.py`, `tokeymeter/cache_admin.py` | `cache_stats()`, `clear_cache()`, `evict_expired()`, `export_cache()`, `import_cache()` | Operational control of the cache. |

---

## 5. Dependency layering (why the structure holds)

The intra-package import graph is acyclic and cleanly layered. Read top-to-bottom
as "depends on what's below it":

```
tokeymeter/__init__.py        (public API: binds implementation symbols,
        │                      then re-exports the curated wrappers)
        ▼
tokeymeter/_api.py            (curated product wrappers: meter, report, …)
        │  composes the implementation modules below
        ▼
tokeymeter/decorator.py             (orchestrator — the hub)
        │  imports: audit, compression, degraded, envelope, memory,
        │           policy, pricing, savings, semantic, storage, utils
        ▼
stage modules                 router · salience · safe_compress · compression ·
        │                     cache_optimize · semantic · memory · storage
        ▼
shared leaves                 pricing · utils · envelope · prompt_ir · events
                              decision → audit/   (decision feeds the ledger)
```

Notable, healthy properties:

- **`pricing.py` is a pure leaf** — depended on by `cache_optimize`, `compression`,
  `router`, `safe_compress`, `salience`; depends on nothing internal. Correct: cost
  estimation is a shared primitive.
- **The compression family clusters** — `salience` and `safe_compress` both build on
  `compression`; `decorator` composes all three. No cycles.
- **`audit/` and `backends/` are self-contained subpackages** with clean internal
  `__init__.py` surfaces — exactly the boundaries you'd draw by hand.
- **`decorator.py` is the single hub.** Everything funnels through it. That is both
  why the single-pass pipeline integrates seamlessly *and* why the file is large
  (§6).

---

## 6. Standing engineering note: `decorator.py`

`decorator.py` is **1,735 lines** — by far the largest module, and the one place the
tree diverges from "every file is small and single-purpose." It earns its size
honestly (it is the orchestrator that wires all seven stages into one pass), but it
is the natural next refactor target: decompose it into per-stage helpers behind a
stable public facade, with the test suite as the safety net, splitting one stage at
a time and re-running `pytest` after each move.

This is deliberately **not** done as part of this map. Reorganizing it touches the
hottest path in the library and the tests import it directly; it deserves its own
focused, test-gated session rather than being bundled into a documentation pass.

---

## 7. "I want to change X — which file?"

| If you want to… | Open |
|---|---|
| Change how cache keys are computed | `tokeymeter/utils.py` |
| Tune exact-cache storage / TTL behavior | `tokeymeter/storage.py`, `tokeymeter/envelope.py` |
| Change semantic-match thresholds or the encoder | `tokeymeter/semantic.py` |
| Adjust routing/cascade decisions | `tokeymeter/router.py` |
| Change compression aggressiveness or add a compressor | `tokeymeter/salience.py`, `tokeymeter/compression.py` |
| Tune the compression safety gates | `tokeymeter/safe_compress.py` (gate), `tokeymeter/decorator.py` (cap + breaker) |
| Add a provider-cache rule or volatile-content detector | `tokeymeter/cache_optimize.py` |
| Change pricing tables or token estimation | `tokeymeter/pricing.py` |
| Change what `report()` returns | `tokeymeter/savings.py` |
| Touch the audit chain / proofs / signing | `tokeymeter/audit/` |
| Change PII redaction rules | `tokeymeter/privacy.py` |
| Adjust fail-open observability | `tokeymeter/degraded.py` |
| Add or change a public API name | `tokeymeter/_api.py` (the wrapper) + `tokeymeter/__init__.py` (export it) |
| Bump the version | `tokeymeter/__init__.py` (`__version__`) **and** `pyproject.toml` |

---

## 8. Honest limitations (so the map doesn't oversell)

- **Numbers are synthetic.** The pipeline is built and tested (530-test suite,
  33-check stress harness), but savings/quality figures are measured on synthetic
  workloads, not yet validated against live providers at scale.
- **The audit chain is per-process by design.** Decisions from other processes can
  stream to a panel/observer but do **not** write into another process's local
  chain. This is intentional, not a gap — but it means a "verified head" must be
  demonstrated from the process that actually recorded the decisions.
- **Quality is *protected + measured + auto-fallback*, not zero-risk.** The four
  compression safeguards make degradation rare and bounded, not impossible.
