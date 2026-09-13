# TOKEYMETER SELF-HOSTED — BUILD DOCUMENT

**Status:** Draft for founder ratification. Everything in §0–§1 freezes on ratification; everything else is working spec.
**Scope:** The self-hosted offering of Tokeymeter. One product, one ledger, spanning API and self-hosted execution. There is no separate "TokenHost" — that fork is permanently vetoed.
**Date:** July 21, 2026

---

## §0 — Positioning (external, freeze on ratification)

> **Tokeymeter is the execution intelligence layer for enterprise AI.**
> It continuously finds, quantifies, and verifies the highest-value improvements in AI execution — on your infrastructure, through your change process, proven in one ledger finance can audit.

The last clause is the sentence no adjacent tool can say. Prometheus sees the cluster and knows nothing about cost. The provider dashboard sees the API and knows nothing about your GPUs. Kubernetes runs workloads and proves nothing. Tokeymeter is the only system that holds API and self-hosted execution in the same books, priced in the same units, verified against itself.

**What Tokeymeter is not, stated plainly in every deck:** not a metrics system (Prometheus stays Prometheus), not a dashboard (Grafana stays Grafana), not a scheduler or orchestrator (Kubernetes stays Kubernetes), not a serving engine (vLLM stays vLLM), not a gateway (no traffic flows *through* Tokeymeter that didn't already flow through the application).

**Economics discipline:** cost savings are a *consequence* of execution intelligence, never the identity. The identity is: measured execution, understood execution, improved execution, proven improvement. The invoice-line outcome is savings; the product is the intelligence and the proof.

**Position on the passive↔active spectrum:** deliberately between passive analytics and active orchestration. More valuable than a dashboard because every recommendation closes its own loop with verification and (within a hard boundary) action. Safely outside orchestration because Tokeymeter never touches infrastructure state — ever. The boundary is defined by *capability*, not by approval workflows (§4).

**Why customers pay (external paragraph, freeze on ratification):** Enterprises running self-hosted inference at scale (typically 8+ GPUs, production traffic) face substantial capacity waste, broken chargeback, and unanswerable board questions on TCO vs. API [market claim — verify & cite before external use]. Tokeymeter closes these gaps with one ledger: finance-grade unit economics, verified recommendations, and demand-side controls that recover capacity without new hardware. Customers pay because it directly protects margins, satisfies auditors, and de-risks multi-million-dollar GPU decisions — outcomes no dashboard or orchestrator can claim.

**Competitive moat (top-tier differentiation, use verbatim in decks):**
1. **The Hybrid Ledger** — the only unified execution record across API and owned hardware. This single capability makes S3 Hybrid Placement irreplaceable. *(SHIPPED)*
2. **Verified Closed Loops** — recommendations are records with evidence windows, predicted floors, and automatic settlement; failed verifications displayed prominently — a trust feature competitors avoid. *(engine: Phase S2 of build)*
3. **Demand-Side Control** — safe in-process actuation (fail-open, canary, auto-revert) that never touches supply. *(primitives shipped; policies phased)*
4. **Calibration Flywheel** — per-customer estimator accuracy that improves with tenure and cannot be exported; new entrants start at zero credibility.
5. **Ritual Integration** — every major output is the named input to an unavoidable business process (monthly cost close, capacity review, procurement event, compliance filing). Products bolted to rituals are not skipped.

**Reach (say this early — it doubles the addressable market):** *"Tokeymeter is for companies that take self-hosted AI seriously. Whether your GPUs live in the cloud, in your own data center, or both — we give you the intelligence, unit economics, and optimization layer that hyperscalers and monitoring tools cannot provide."* Self-hosted here means **you control the serving**, not that you own the metal. A rented H100 fleet on a cloud provider has exactly the same blind spots as an on-prem rack, and the same buyer.

**The layer statement:** hyperscalers provide the GPUs; Kubernetes and vLLM provide the execution; **Tokeymeter provides the missing intelligence and control layer that makes self-hosting economically superior and strategically defensible.** We do not replace infrastructure — we make it legible, and then we make it better.

**Taglines (for decks):**
- "One ledger. Real execution intelligence. Proven improvements."
- "Self-hosted AI that pays for itself — measured, verified, optimized."
- "The missing intelligence layer between your GPUs and your P&L."


### §0.0 — How it delivers, and the outcome (the short external pitch)

**How it delivers — four moves:**
1. **Unified ledger + unit economics.** Joins the customer's *existing* telemetry to produce accurate $/1K-token costs on their hardware, side by side with any API provider. **[SHIPPED]**
2. **Verified recommendations.** Deterministic detectors surface high-ROI opportunities — capacity recovery, consolidation, hybrid placement, deferral — each with evidence, a predicted floor, and automatic settlement after the customer acts. **[SPEC — Phase S2]**
3. **Safe optimization.** Demand-side policies in-process (fail-open, canary, auto-revert) for routing, priority shedding, budget guards, and continuity; **change artifacts** for anything infrastructural, flowing through their own GitOps/PR process. **[primitives PARTIAL; policy set SPEC]**
4. **Closed-loop proof.** Every recommendation is verified. Successes *and failures* are visible. The system calibrates to the customer's environment over time. **[SPEC — §3.9/§3.10]**

**The outcome for self-hosters:** you move from *"we think we're saving money"* to **proven, finance-auditable execution intelligence** — measurable capacity gains, proper chargeback, de-risked procurement, and compliance-ready reporting. Tokeymeter doesn't replace infrastructure; it makes self-hosted AI **smarter, more efficient, more reliable, and strategically defensible**, while providing **independent truth no hyperscaler or monitoring tool can match**.

> **Naming guardrail (standing decision):** the "AI factory" framing stays **internal**. External copy uses concrete outcomes — capacity recovered, unit cost, flip threshold, chargeback — because the concrete version outsells the metaphor with exactly the buyers who write the cheque.

### §0.3 — Against the alternatives (the "why not Kubecost / CloudZero / Vantage / Finout / hyperscaler tools?" answer)

Every buyer asks this, and the honest answer is not "they are bad" — it is that they answer a *different question*. They report cost; we prove execution economics and close the loop. The distinction to lead with: **they give visibility; we give verified execution advantage.**

| Their question | Kubecost / CloudZero / Vantage / Finout / hyperscaler cost tools | Tokeymeter |
|---|---|---|
| True unit cost **on your own GPUs**? | Limited — built for cloud/K8s billing and allocation; self-hosted GPU-second economics is thin or absent | **$/1K tokens on your hardware with shown derivation** *(SHIPPED)* |
| Self-hosted **and** API in **one priced, reconciled** view? | Weak or none — cloud/API-centric | The **only unified hybrid ledger**, per-model reconciled *(SHIPPED)* |
| Do the recommendations **prove they worked**? | Suggestions and dashboards; no settlement | **Settlement query + automatic verification; failures shown** *(engine: Phase S2)* |
| Can it **act safely without touching infrastructure**? | No demand-side actuation | **In-process policies: fail-open, canary, auto-revert** *(primitives PARTIAL; policies SPEC)* |
| Does removal **break a process**? | Usage-based tool; swappable | **Finance close runs on the ledger; calibration history can't be exported** |

**What each is genuinely good at, so we never misrepresent them** (a buyer who owns Kubecost will not trust a deck that pretends it does nothing): Kubecost — Kubernetes cost allocation and right-sizing. CloudZero — engineering-led cloud cost intelligence. Vantage / Finout — multi-cloud FinOps visibility and reporting. Hyperscaler tools (AWS Cost Explorer, Azure Cost Management, Google Recommender) — billing and basic optimization inside one provider's ecosystem. **All four categories are cloud-bill-centric, none holds owned GPU execution and API usage in one reconciled ledger, and none closes a verified loop.** That gap is the whole product.

**We complement, we do not replace.** These tools — and Kubernetes, Prometheus, Grafana, vLLM — keep doing their jobs. We sit above them, join their data read-only, and add the layer none of them has: hybrid unit economics, verified closed loops, and finance-auditable proof. In a security review this is also the safety story (§4): we are not a competitor to the orchestrator, we are demand-side intelligence on top of it.

**The one-line objection handler:** *"When finance asks 'show me the real cost and prove it improved,' or procurement asks 'why this many GPUs?', only Tokeymeter answers with a single auditable source of truth and verified outcomes. The others stop at the dashboard."*

### §0.1 — What Tokeymeter Improves (the five dimensions)

The answer to the first question every buyer asks. Each dimension is marked **[SHIPPED]** (working code today), **[SPEC]** (designed in this document, not yet built), or **[PARTIAL]**. *Nothing in this list may be presented to a customer as available unless it is marked SHIPPED — see §11.*

**1. Capacity utilization & efficiency** *(the biggest day-to-day win)*
Recovers wasted GPU time — idle windows, poor batching, cache misses, duplicate deployments **[cache + single-flight recovery SHIPPED via S1; batching/idle/dedup detectors SPEC]**; defers deferrable work to off-peak **[SPEC — §3.7]**; optimizes queue behavior and right-sizing **[SPEC]**.
→ **Result: higher effective utilization without buying more GPUs.** Market anchors put recoverable capacity in the 15–40% range `[market claim — verify & cite before external use; our number is always the customer's own settled recovery]`.

**2. Unit economics & cost control**
Accurate **$/1K tokens and $/request on your own hardware, with full provenance** **[SHIPPED — S1]**; chargeback/showback to teams and features **[SHIPPED — S2]**; identifying the cheapest valid endpoint per request class **[evidence SHIPPED via S3; automated routing SPEC — §3.7]**.
→ **Result: lower true cost per unit of AI work, and transparent allocation.**

**3. Decision quality** *(strategic)*
Data-driven build-vs-buy and hybrid placement **[SHIPPED — S3]**; procurement evidence packs **[SPEC — S8]**; explicit thresholds — *"at X% utilization, self-hosting beats the API"* **[SHIPPED — `flip_utilization`]**.
→ **Result: better capital allocation and renewal decisions instead of vibes-based procurement.**

**4. Reliability & continuity**
Detects serving degradation from your own request stream **[SPEC — S11]**; safe failover between declared endpoints with automatic return **[SPEC — S11]**; protects revenue traffic via priority shedding and budget guards **[primitives PARTIAL; policies SPEC — S12]**.
→ **Result: fewer customer-facing incidents and measurable continuity during outages.**

**5. Trust, governance & compliance**
One finance-auditable ledger with provenance on every number **[SHIPPED]**; verified wins **and failed recommendations both displayed** **[verification harness SPEC — Phase S2]**; CSRD/energy attribution **[SPEC — S9]**; accumulated execution history and estimator calibration **[SPEC — §3.10]**.
→ **Result: stronger audit posture, board confidence, and durable institutional knowledge.**

**The big picture.** Tokeymeter turns self-hosted AI from a **cost center with blind spots** into a **measured, optimized, and proven capability** — improving *how efficiently* the GPUs are used, *how transparently* costs are understood and allocated, *how intelligently* placement and procurement decisions get made, *how reliably* AI features perform under pressure, and *how defensibly* execution can be reported and governed.

**One line:** *we improve execution quality, economic clarity, operational safety, and strategic confidence for self-hosted AI.* Customers pay not for reports, but for **measurable, verifiable improvements** in how their infrastructure performs and how confidently they can manage it.

### §0.2 — Benefits by stakeholder (deck-ready)

| Stakeholder | What they lose sleep over | What Tokeymeter gives them | Status |
|---|---|---|---|
| **CFO / FinOps** | "We spend heavily on GPUs and cannot allocate a dollar of it." | Unit cost with derivation; per-team chargeback in the same units as the API bill; a close that reconciles | **SHIPPED** (S1, S2) |
| **AI Platform / Infra lead** | "I know we're wasting capacity, I can't enumerate it, and I have no budget for more GPUs." | Recoverable capacity by mechanism, with evidence; safe demand-side optimization; continuity | S1 **SHIPPED**; policies + continuity **SPEC** |
| **CTO / Board** | "Is self-hosting still the right call, and can I defend it?" | Build-vs-buy evidence, the utilization flip threshold, procurement packs | S3 **SHIPPED**; S8 **SPEC** |
| **Sustainability / Compliance (EU)** | "We must report AI energy per workload and cannot." | Per-workload energy attribution with derivation, CSRD-formatted | **SPEC** (S9) |
| **Security / Risk review** | "What does this thing touch, and can it take us down?" | The §4 boundary table; fail-open guarantee; content-blind by construction | **SHIPPED** (verified under kill tests) |

**Sales rule for both tables:** lead with the stakeholder's own pain in their own words, show the artifact, then state plainly what is shipped and what is roadmap. **The status column is not a weakness in the deck — it is the reason the shipped claims get believed.**

---

## §1 — The Non-Negotiables

Print these in the engineering handbook. Every feature, PR, and sentence of copy is checked against this list.

1. **Tokeymeter never replaces infrastructure. It continuously improves it.** Infrastructure remains responsible for execution; Tokeymeter remains responsible for intelligence.
2. **Direct action only inside the process boundary Tokeymeter already occupies.** Everything that would touch infrastructure ships as a change artifact the customer applies through their own change process. Approval is not the boundary; capability is.
3. **No unverifiable recommendation ships.** If we cannot specify, in advance, the ledger query that will prove or disprove it, it is an opinion, and opinions do not go in the product.
4. **No fabricated numbers.** Every figure carries provenance (measured / declared / estimated, with derivation shown). Estimates are labeled as estimates everywhere they appear, including in sales material. The honesty block is mandatory on every report.
5. **One product, one ledger.** API and self-hosted execution live in the same ledger with the same schema. The hybrid view is the moat; forking it destroys the moat.
6. **Tokeymeter ingests telemetry; it never collects infrastructure metrics.** No exporters, no agents on nodes, no scraping infrastructure we own. We join the customer's *existing* metrics (their Prometheus, their DCGM) to our execution ledger, read-only.
7. **Content-blind everywhere.** Nothing in the self-hosted offering reads, stores, or requires prompt/response content. Metadata and measurements only.
8. **Compliance is an output, not the product.** Energy/CSRD, audit, chargeback are reports the ledger emits — never the pitch lead.
9. **External copy states concrete outcomes.** The internal spine (Measure → Explain → Optimize → Prove → Improve) stays internal. No "AI Factory" language in customer-facing material.
10. **Failed verifications are shown, not hidden.** A settled-failed recommendation is displayed with the same prominence as a settled-verified one, and it recalibrates the estimator. This is a trust feature and a sales weapon.

---

## §2 — The Buyer and the Pains

### Who buys, and when

Self-hosted AI inside an enterprise has five stakeholders, and each one arrives with a *trigger event*, not a general interest in optimization. We sell against triggers.

| Stakeholder | Bleeding pain | Trigger that opens the budget |
|---|---|---|
| Platform / infra lead | Cluster is congested at peak, idle off-peak; no defensible picture of where capacity goes | Capacity crunch with no budget for more GPUs |
| Engineering leadership | Cannot answer "is self-hosting this model still worth it?" with evidence | Board / CTO review questioning the self-hosting strategy |
| Finance / FinOps | Shared cluster, zero unit cost, chargeback impossible | Finance mandates chargeback or cost allocation for AI |
| Procurement | Renewal and expansion decisions made on vibes | GPU lease renewal or expansion request lands on the desk |
| Sustainability / compliance (EU) | AI energy footprint must be reported and nobody has the numbers | CSRD reporting cycle |

### Primary buyer profile (add to every deck)

**Primary:** Platform/Infra leads and FinOps leaders in organizations with 20+ GPUs running production self-hosted inference or training. **Secondary:** CTO/Board during renewal or strategy reviews; Sustainability leads in the EU. They buy when triggers hit: capacity crunch with no budget, finance demanding chargeback, board questioning self-hosting economics, or compliance deadlines. We win by arriving with a pilot that proves pains (or narrows scope) using **their own data** in weeks 1–2.

**Trigger timing (tie the sales motion to the calendar):** GPU lease renewals (commonly Q2/Q4), **commitment true-up and mid-term review dates on reserved/committed-use contracts**, CSRD/reporting cycles, quarterly board reviews, budget planning season. S8 Procurement Intelligence exists to be in the room before the renewal date.

### The pain inventory

> **Evidence discipline for this table:** each pain below is paired in sales material with (a) market evidence — external, cited, marked `[market claim]` until verified — and (b) *the customer's own numbers from the week-1–2 pilot instrument (§10)*, which always outrank the market claim. Typical market anchors: low GPU utilization is widespread; shared clusters break FinOps attribution; TCO is commonly underestimated [market claims — verify & cite before external use]. Why they pay: capacity recovery defers GPU buys; chargeback prevents finance from killing self-hosting; placement evidence de-risks board decisions.

These are the structural pains of self-hosted AI. They are well-established patterns in the market, **but honesty rule #4 applies to sales too: we do not claim a pain is validated at a customer until their own ledger shows it.** The validation instrument is part of the product (§10) — every pilot spends its first two weeks proving or disproving these pains with the customer's own data. A disproven pain narrows the scope and *raises* trust.

- **P1 — Wasted execution capacity.** GPUs congested at peak and idle off-peak; caches missing; duplicate work re-executed; deferrable work running at the worst hour. Nobody can enumerate the waste, so nobody can recover it.
- **P2 — No unit cost, therefore no chargeback.** API calls have a price per token; self-hosted requests have nothing. Shared clusters cannot allocate cost to teams, so finance either guesses or gives up. This blocks chargeback, blocks showback, and poisons every downstream decision.
- **P3 — Build-vs-buy anxiety.** The permanent question: at current utilization, is running our own model cheaper than the API alternative — and at what utilization does the answer flip? No tool answers it because no tool sees both sides.
- **P4 — Model sprawl and duplication.** Multiple teams independently running under-utilized copies of the same or overlapping models. Endemic in organizations where teams provision their own serving.
- **P5 — Queue pain and SLO over-provisioning.** Latency targets set once and never revisited; capacity provisioned for a p99 nobody re-examined; queue wait invisible as a business cost.
- **P6 — Procurement on vibes.** "We need more GPUs" backed by a screenshot, not by an execution history. Renewals signed without evidence of what the current fleet actually produced.
- **P8 — Prepaid and committed capacity with unproven ROI.** The organisation has already signed for capacity — reserved instances, committed-use discounts, a prepaid GPU-hour block, or a multi-year lease. The money is *spent or contractually owed*, and nobody can say what fraction of that commitment is actually being consumed, or what the *effective* rate is once under-consumption is priced in. This is distinct from P1 (recoverable waste, happening now) and P6 (a future purchase decision): here the decision is already made and the question is ROI on money that is gone. It carries its own calendar — mid-commitment reviews, true-up dates, renegotiation windows.
- **P7 — Energy and CSRD exposure (EU).** AI energy attribution required for reporting; nobody has per-workload numbers. Compliance budgets get spent even when optimization budgets don't.

### Pain validation matrix *(sales-facing — carry this table into every deck)*

| Pain | Market evidence `[market claim — verify & cite before external use]` | Why customers pay | Service |
|---|---|---|---|
| **P1** Wasted capacity | GPU utilization commonly reported in the 15–40% range; idle time dominates bills | Direct capacity recovery **defers GPU purchases** — the budget line a CFO already feels | S1, S13 |
| **P2** No unit cost / chargeback | Shared clusters break FinOps attribution; API spend is allocatable, owned hardware is not | Enables showback/chargeback; **prevents a finance mandate from killing self-hosting** | S2 |
| **P3** Build-vs-buy anxiety | Self-hosting TCO frequently underestimated (reported ranges up to 40–165%); break-even rarely computed | Data-driven placement decisions; **quantifies hybrid arbitrage** with a defensible threshold | S3, S10 |
| **P4** Model sprawl | Teams provision independently; duplicate under-utilized deployments are endemic | Consolidation recovers capacity **without new hardware** | S4, S14 |
| **P5** Queue pain / SLO over-provisioning | Latency targets set once, never revisited; queue wait invisible as a cost | Right-sizes the SLO; **turns an engineering guess into a priced decision** | S5, S7 |
| **P6** Procurement on vibes | Renewals signed without evidence of what the fleet produced | **De-risks a multi-million-dollar signature** — and the trigger has a date | S8 |
| **P8** Prepaid / committed capacity | Reserved instances and committed-use discounts are standard in cloud GPU purchasing; commitment *utilisation* is rarely measured | **Proves ROI on money already committed**, and turns the next true-up or renegotiation into an evidence-backed conversation | S1, S3, S8 |
| **P7** Energy / CSRD exposure (EU) | Per-workload AI energy attribution required, rarely available | Unlocks a compliance budget for work that also cuts cost | S9 |

**Reading rule for this table:** the middle column is *external market evidence and must be cited before external use* — it opens the conversation. The customer's own week-1–2 pilot numbers (§10) close it, and always outrank the market claim. **Never present a market percentage as our measurement (§11).**

Each pain maps to a paid service in §5. Each service is designed as the named input to a *recurring business ritual* (§6), which is what makes the product unskippable rather than nice-to-have.

---

## §3 — Architecture

```
Application processes  (Tokeymeter in-process — SHIPPED)
        │  every request → evidence
        ▼
EXECUTION LEDGER  (shipped; extended with self-host fields, §3.1)
        ▲                                  ▲
        │ read-only joins                  │ declared inputs
Telemetry Ingestion (§3.2)         Declared Cost Model (§3.3)
(customer's Prometheus/DCGM,       (register_cluster_costs:
 serving metrics, scheduler         capex, lease, power —
 logs — we ingest, never collect)   derivation shown)
        │
        ▼
UNIT-COST ENGINE (§3.4)  →  $/1K tokens on YOUR cluster, comparable to API pricing
        │
        ▼
OPTIMIZATION ENGINE (§3.5)
  Detectors → Ranking → ROI Estimation → RECOMMENDATION RECORD (§3.6)
        │
        ├──► IN-PROCESS POLICY ACTUATOR (§3.7)   [bounded, auto-revert]
        │
        └──► CHANGE ARTIFACT EMITTER (§3.8)      [PR, never push-deploy]
                       │
                       ▼
VERIFICATION HARNESS (§3.9)  →  settled-verified / settled-failed
                       │
                       ▼
CALIBRATION STORE (§3.10)  →  per-customer estimator accuracy (moat)
```

No scheduler. No orchestrator. No gateway. No collector. **Closed-loop optimization with a hard capability boundary.**

### §3.1 Ledger extensions for self-hosted execution

Per-request fields added to the existing schema (all content-blind):

- `endpoint_identity` — which serving endpoint (customer-declared name), which model, which deployment
- `queue_wait_ms` — time between submission and execution start, where the serving layer reports it; estimated-and-labeled where it doesn't
- `tokens_provenance` — `reported` vs `estimated`, per record. **This closes the known shadow-hit bug** (records logging estimated tokens despite upstream reported usage); the fix ships in Phase S0, not later.
- `gpu_occupancy_attribution` — GPU-seconds attributed to the request: measured where serving metrics expose it, estimated-with-stated-method where they don't, labeled either way
- `batch_context` — whether the request executed batched, and batch size class
- `energy_attribution` — optional, derived from occupancy × declared node power draw; always labeled estimated

### §3.2 Telemetry Ingestion (read-only, never collection)

Adapters that *query the customer's existing telemetry* and join it to the ledger on time-window + endpoint + model:

- Prometheus HTTP query API (their Prometheus, their exporters)
- NVIDIA DCGM exports / nvidia-smi snapshot files
- vLLM / TGI native metrics endpoints (both expose Prometheus-format metrics)
- Optional: cloud/lease billing exports as inputs to the cost model

Rules: read-only; we store joined aggregates, never raw infrastructure time-series (Prometheus remains the system of record for metrics); absence of any telemetry source **degrades output to estimate-labeled, never blocks** — the same graceful-degradation posture already shipped in the engine.

Strategic effect: in every security review and every "aren't you Prometheus?" conversation, the answer is structural — *we don't collect metrics, we join yours to the only execution ledger you have.*

### §3.3 Declared Cost Model

`register_cluster_costs(...)` — the self-hosted sibling of the shipped `register_pricing`:

- Inputs: hardware capex + depreciation schedule, **or** lease $/month, **or a prepaid/committed contract** (committed amount per period + the capacity that commitment entitles you to); power rate; facility overhead factor; support/staff allocation (optional)

**Committed capacity carries a utilisation dimension, and ignoring it understates cost.** On a commitment you pay for the entitled capacity whether or not you consume it, so the *effective* rate is the sticker rate divided by the fraction actually consumed: commit to N GPU-hours, consume half, and your true rate is 2x sticker. The model therefore derives and shows BOTH:
  - `committed_rate` — the contractual rate, and
  - `effective_rate` — committed spend ÷ capacity actually consumed in the period, with the consumed fraction (`commitment_utilisation`) shown alongside.

**This is a correctness requirement, not a nicety.** Reporting the sticker rate on an under-consumed commitment understates self-hosted unit cost, which propagates into §3.4 and can flip an S3 build-vs-buy verdict the wrong way — the one direction our honesty rules never tolerate (§1.4). Where the entitled capacity is not declared, `effective_rate` is omitted and stated as unavailable; it is never guessed.
- Output: amortized **$/GPU-second** per node class, with derivation shown exactly like the v0.13 pricing registry shows its derivation
- Provenance stamped on every priced record; honesty block on every report distinguishing *declared* from *measured* from *estimated*

No number in the system exists without a stated origin. This is the July 18 DNA — "savings a CFO could audit" — extended to owned hardware.

### §3.4 Unit-Cost Engine

Produces, per request and per aggregate: **$/request and $/1K tokens on the customer's own cluster** = attributed GPU-seconds × amortized $/GPU-second (+ declared overheads), derivation shown.

And the column that changes conversations: the same request priced against **API list alternatives** from the already-shipped pricing registry, side by side, same units. This single table unlocks chargeback (P2) and feeds arbitrage (P3). Nobody else can print it, because nobody else has both sides in one ledger.

**The API side must model token *types*, or the comparison is wrong for modern models.** Contemporary API pricing is not one input rate and one output rate: it distinguishes **cached-input tokens** (priced well below fresh input) and **reasoning tokens** (billed like output but never appearing in the visible response). A comparison that prices only plain input+output *understates* the API cost of a reasoning-heavy workload and can flip an S3 verdict toward the API incorrectly — the same class of correctness risk as the committed-rate issue in §3.3. The pricing registry therefore accepts per-type rates where a provider exposes them (cached-input, reasoning/thinking, plus standard input/output), each type carried on the record with its own provenance; where a provider does not break them out, the comparison states that it used the plain input/output rate and is a lower bound on API cost. **Never silently price reasoning tokens as ordinary output, and never invent a cached-token ratio the customer did not supply.**

**Added outputs (ratified):**
- **Per-aggregate "Effective $/1K tokens"** with *confidence bands drawn from the calibration store* (§3.10) — the band is empty and stated as such until enough settled history exists; never a fabricated interval.
- **"Remaining headroom before API crossover"** — the utilization % threshold at which self-hosting stops winning, plus a *projected crossover date* derived from the customer's own growth trend in the ledger. The projection prints its trend window, its method, and the fact that it is a projection. This is the single number that converts a philosophical build-vs-buy argument into an operations target. *(Threshold shipped in S3 as `flip_utilization`; the dated projection is the extension.)*

### §3.5 Optimization Engine

**v1 is deterministic. No ML claims, no "learns from every request" language until a learning system exists.** What v1 honestly does:

- **Detectors:** pure functions over the ledger. Each detector = evidence window + threshold + emitted opportunity record. Auditable, explainable, testable.
- **Ranking:** estimated annualized value × confidence class.
- **ROI estimation:** every estimate carries its assumptions inline and a *predicted floor* (the minimum delta below which the recommendation is judged failed at settlement).
- **Calibration (§3.10):** estimator parameters recalibrate against settled outcomes. This is learning in the accounting sense — mechanical, honest, and true on day one.

Launch detectors (Phase S2): idle-window capacity (P1), duplicate-deployment consolidation (P4), cache-opportunity (P1), off-peak deferral candidate (P1/P5), hybrid placement threshold breach (P3).

**Detector prioritization (ratified):** the launch set above is ordered by *high-ROI-first* — idle-window reclamation, duplicate model consolidation (with migration artifact), metadata-driven batching opportunity, and hybrid threshold breach lead, because each maps to a pain with a budget attached.

**New detector — Anomaly & Regression Detector.** Flags sudden utilization drops, queue-wait spikes, and cost-per-request increases, settled against a historical baseline from the same ledger. It is the proactive input to a customer ritual (weekly engineering sync) rather than a passive chart. Same discipline as every other detector: deterministic, evidence-windowed, with a predicted floor and a settlement query. **No ML claim** — a baseline comparison is a baseline comparison.

**Ranking gains a third factor — Time-to-Value.** Rank = estimated annualized value × confidence class × time-to-value class (policy changes settle in days; change artifacts settle in weeks). A smaller win that settles this week often outranks a larger one that settles next quarter, because settled wins are what renew contracts.

### §3.6 Recommendation Record (schema)

Every recommendation is a first-class ledger object:

```
recommendation:
  id, detector_id, created_at
  evidence_window: {from, to, ledger_query}
  finding: human-readable, numbers with provenance tags
  predicted_delta: {metric, estimate_range, floor, assumptions[]}
  settlement: {query, window, compare_method}
  action_class: IN_PROCESS_POLICY | CHANGE_ARTIFACT | EVIDENCE_ONLY
  artifact_ref: (diff / policy spec / report section)
  status: proposed → approved → applied/merged → settled_verified | settled_failed | reverted
  settled_result: {measured_delta, vs_floor, notes}
  business_impact_estimate: {currency_delta_annualized, confidence,
                             affected_stakeholders[]}   # e.g. "Finance: chargeback",
                                                        #      "Procurement: renewal"
  ritual_link: <the recurring process this feeds>       # e.g. "Monthly Cost Close",
                                                        #      "Q3 Procurement"
```

**Why the two new fields exist.** `business_impact_estimate` names *who in the customer's org cares* — a recommendation nobody owns never gets applied, and an unapplied recommendation never settles. `ritual_link` is the moat made structural: it forces every recommendation to declare which unavoidable meeting it feeds. **A recommendation with no ritual_link is a dashboard insight, and dashboard insights are what we refuse to build (§9).**

The lifecycle **Observe → Measure → Understand → Recommend → Approve → Act → Verify → Learn** is not a diagram; it is this record's state machine.

### §3.7 In-Process Policy Actuator — Demand-Side Execution Management (the active surface)

**The doctrine that sizes the active surface: infrastructure owns supply; Tokeymeter owns demand.** Kubernetes, vLLM, and the schedulers decide what serves requests. Tokeymeter — because it already lives inside the process that *issues* requests — decides what is asked, when, where among declared endpoints, at what priority, and under what budget. Demand-side management can be aggressively active without ever touching infrastructure state, and it is where the paid active capabilities live.

Upon customer approval, Tokeymeter may directly apply, within the process it already runs in:

- **Execution Continuity (active failover).** Detects endpoint degradation from its own request stream (latency, error rate, queue wait — no external telemetry required), shifts routing weight to declared fallbacks (another self-hosted endpoint or a declared API model), returns automatically on recovery, and prints the continuity report: incident window, requests protected, cost delta of the failover period.
- **Priority shedding (peak protection).** When queue waits breach a declared SLO, applies priority policy keyed on request metadata (team, feature tag, request class — never content): defer, downgrade, or park low-priority classes to protect revenue-facing traffic. Demand is shaped at the source; supply is never touched.
- **Budget guards (spend circuit breakers).** Declared budgets per team/feature; on breach: throttle, downgrade to a cheaper declared model, or block with an explicit, ledger-logged error. (Primitives live here; org-wide policy governance is TokeNet's plane — same ledger, clean upsell seam.)
- **Metadata-keyed model arbitrage.** Routes customer-defined eligible request classes to cheaper declared models/endpoints under approved policy. Eligibility is declared by metadata, never inferred from content.
- **Off-peak release windows.** Application-tagged deferrable work is held in-process and released in declared windows. Tokeymeter times its own requests — client-side release, not cluster scheduling.
- **Optimization primitives.** Cache TTL/scope/admission, compression levels, batching hints — the original surface.

**New policies (ratified additions):**

- **Dynamic Model Routing Guard.** Metadata *plus real-time queue and cost signals* route eligible traffic to cheaper declared endpoints (self-hosted fallback or a declared API model) while respecting declared latency SLOs. Distinct from static metadata arbitrage above: the routing decision reads live pressure, not just the request's tag. Settles A/B-style against the control slice. Eligibility remains declared by metadata — **never inferred from content**.
- **Workload Deferral Scheduler (in-process).** Application-tagged deferrable requests (e.g. batch analytics) are held in-process and released in declared off-peak windows, with retry logic and a hard maximum hold time. Prints an **"avoided peak cost"** report. This is Tokeymeter timing *its own* requests — not cluster scheduling, not a queue service.
- **Auto-Canary for New Deployments.** On a declared model-version or deployment change, shift traffic fraction gradually and **auto-revert on regression** across latency, error rate, or unit cost. Turns every deployment into a settled experiment instead of a hope.

All three inherit the invariants below without exception: fail-open, canary-first, auto-revert, ledger-logged.

**Every policy rolls out as a canary.** Apply to a declared traffic fraction, settle against the control slice, then ramp or revert. Settlement thereby becomes controlled-experiment evidence, not before/after inference.

Every applied policy carries: predicted floor, settlement window, and an **auto-revert condition** — if the verified delta undershoots the floor, the policy reverts itself and logs the reversion. And one invariant over the entire active surface: **all policies fail open.** If Tokeymeter dies, every policy vanishes and traffic flows on default paths — execution never depends on Tokeymeter being alive. The customer cannot be silently made worse off, and cannot be taken down by us. Both guarantees go in the sales deck.

### §3.8 Change Artifact Emitter (the "active-shaped passive" surface)

For everything infrastructure-touching, the recommendation *is* the change, emitted as an artifact the customer applies through their own pipeline:

- vLLM/TGI flag and config diffs (e.g., batching, quantization, max-num-seqs changes)
- Helm values / deployment manifest diffs (e.g., consolidation plans, replica changes)
- cron/batch-window definitions for deferral
- procurement evidence packs (§5, S8)

Each artifact ships with its evidence pack and its settlement query pre-attached. **Pull request, never push deploy.** It flows through the customer's change management — which enterprises prefer — and the loop still closes, because verification fires after merge regardless of who merged.

### §3.9 Verification Harness

Pre/post comparison over the same ledger that produced the recommendation: baseline window vs settlement window, method declared in the record (like-for-like traffic normalization where volume shifted, stated when applied). Outputs `settled_verified` or `settled_failed`. **Failed settlements are displayed, feed calibration, and appear in the customer's monthly report.** A vendor that shows its misses is the only vendor whose hits are believable.

### §3.10 Calibration Store

Per-customer record of predicted-vs-settled deltas per detector. Two jobs: (1) tighten estimate ranges over time; (2) print the **estimator accuracy report** — "over the last quarter, our capacity predictions settled within X of estimate" (X computed, never asserted). This store is per-customer operational state that cannot be exported to a competitor. It is the flywheel half of the moat (§6).

### §3.11 Advisory Signals for Customer Control Loops

The clean bridge between our demand intelligence and their supply control: Tokeymeter **publishes** read-only signals — demand forecast, queue-pressure index, deferral backlog — as metrics the customer's own autoscaler (KEDA, HPA) can consume if *they* wire it. We never write to a scaler, never trigger a scale event, never know whether the signal was consumed. Their control loop, our intelligence feeding it. This gives customers supply-side reactivity powered by Tokeymeter without Tokeymeter ever crossing the boundary in §4.

**Added signals (ratified):**
- `capacity_pressure_forecast` — short- and medium-term pressure derived from ledger trends, published read-only with its trend window and method stated.
- `recommended_slo_adjustment` — read-only suggestion where queue economics show an SLO is either unaffordable or needlessly expensive.

Both are **published, never pushed.** This is the positioning line for the deck: *Tokeymeter becomes the intelligence feed for the customer's existing HPA/KEDA autoscalers and FinOps tools — their loops, our data.*

### §3.12 Enterprise Integration & Export Layer *(new)*

Adoption friction and ritual embedding are the same problem. This layer exists so an output lands **inside the customer's existing process** without asking them to adopt a new one.

**Standardized exports.**
- **Finance:** CSV/GL-shaped chargeback feeds for the monthly close, including **direct ERP/GL import formats** (the finance team should not have to reshape our output before it enters their system of record) *(S2 `chargeback_csv` shipped — formula-injection-safe, RFC-4180, reconciling by construction)*.
- **Procurement:** evidence packs bundling TCO models, flip thresholds, and settled recovery history.
- **Compliance:** CSRD-formatted energy and provenance output *(byproduct, never the lead — §0)*.

**Ritual hooks.** Webhook and OpenTelemetry hooks: post the monthly artifact to Slack/Teams as a review-agenda item, open a Jira ticket when a change artifact is emitted, attach the settlement result when it closes. **OTel/tracing ingestion also carries request metadata (team/feature tags) where the customer already emits it — reducing or eliminating code changes for attribution.** Read-only ingestion, per §3.2: we consume their traces, we never become their tracing system.

**Audit log export.** Every recommendation and every settlement, exportable for external assurance. This is what makes the calibration history a *defensible record* rather than an internal claim — and it is a large part of why replacing Tokeymeter is expensive (§6).

**Pre-built adapters (frictionless adoption).** Kubernetes + vLLM/TGI deployment patterns, common scheduler conventions, and major cloud **lease/invoice export formats** so the Declared Cost Model (§3.3) can be populated from documents finance already produces instead of a questionnaire.

---

## §4 — The Action Boundary (print this table in every security review)

| | Tokeymeter MAY touch (in-process only) | Tokeymeter NEVER touches (change artifacts only) |
|---|---|---|
| Scope | **Demand-side, in-process:** failover routing among declared endpoints · priority shedding by request metadata · budget guards · metadata-keyed model arbitrage · off-peak release of tagged work · cache/compression/batching primitives · publishing read-only advisory signals | **Supply-side:** pods, nodes, schedulers, autoscalers · serving-engine configs & restarts · model weights & placement · network, storage, IAM · anything requiring kubectl/terraform/helm to take effect |
| Preconditions | Customer-approved policy · predicted floor · settlement window · auto-revert armed | Emitted as diff + evidence pack + settlement query; applied by the customer through their own change process |
| Failure posture | Auto-revert + logged reversion | Verification fires after merge; settled-failed displayed and calibrated |

One sentence version of the whole table: **infrastructure owns supply; Tokeymeter owns demand.** Everything we do actively is a manipulation of requests we already carry — what is asked, when, where among declared endpoints, at what priority, under what budget — and all of it fails open.

The boundary is **capability, not approval**. No quantity of approvals turns Tokeymeter into a system that restarts a pod. This table is simultaneously the legal defense in enterprise review and the positioning defense in every "so you're an orchestrator?" conversation.

**One-sentence sales defense (memorize verbatim):** *"Infrastructure owns supply and state. Tokeymeter owns demand intelligence and request shaping. Everything active lives inside the process boundary we already occupy, or is emitted as a customer-applied artifact. That is why we are safe, auditable, and fundamentally different from an orchestrator."*

---

## §5 — Execution Intelligence Services (the paid layer)

Not "modules" — **Execution Intelligence Services**: one platform, one ledger, many outputs. Each service below is specified as: pain → evidence → output → verification → trigger → ritual it feeds. Ordering is deliberate: S1–S3 are the revenue spearhead.

### S1 — Capacity Intelligence *(flagship; first sellable artifact)*
- **Pain:** P1. **Trigger:** capacity crunch, no budget for GPUs.
- **Evidence:** ledger occupancy attribution + ingested utilization + queue waits.
- **Output:** the **Capacity Recovery Report** — enumerated recoverable execution capacity by mechanism (cache, dedup/single-flight, batching, off-peak deferral, right-sizing), each line with provenance and an action class (policy or artifact). "Recover execution capacity," not "monitor GPUs" — execution capacity includes cache, queue, batching, residency, duplication, which is why this never reads as monitoring.
- **Verification:** each recovery line settles individually as its action lands.
- **Ritual:** monthly capacity review; the report *is* the agenda.
- **Commitment coverage (P8):** where a prepaid/committed contract is declared, the report states what fraction of the entitlement was consumed and expresses recovered capacity **against the committed base** — "you are consuming X of what you already pay for; here is the recovery that closes the gap before you buy more."
- **Added outputs (ratified):** a **Recoverable Capacity Heatmap** (where the waste actually sits — by endpoint, model, team, time-of-day) plus a **prioritized action plan carrying an effort/ROI score per line**, and an **Avoided Procurement Value** projection — recovered capacity translated into GPUs-not-bought, stated with its assumptions and marked a projection. This is the line that survives contact with a CFO: not "you saved compute," but "you deferred N GPUs, here is the derivation."

### S2 — Unit-Cost & Chargeback Ledger
- **Pain:** P2. **Trigger:** finance mandates chargeback/showback.
- **Evidence:** unit-cost engine (§3.4) + shipped per-team attribution (already live: team-level attribution demonstrated July 18).
- **Output:** per-team, per-feature, per-model $/period on owned hardware, same schema as API spend, exportable to finance. Honesty block distinguishing measured/declared/estimated.
- **Verification:** reconciliation posture — totals tie to declared cost model, token counts tie to reported usage (the July 18 token-exact discipline, applied monthly).
- **Ritual:** monthly cost close. Once chargeback runs on this ledger, removing Tokeymeter breaks finance's books. This is the deepest unskippability hook in the product.
- **Added outputs (ratified):** team / feature / model views with **drill-down to individual high-cost request patterns (metadata only — never content)**, and an auditor-ready **Reconciliation Report** that ties exactly to the declared cost model *and* to ingested Prometheus aggregates. *(Shipped foundation: `chargeback_report` + `chargeback_csv` — per-row provenance, reconciliation by construction, formula-injection-safe export, malformed records excluded whole and counted.)*

### S3 — Hybrid Placement Intelligence *(the crown jewel)*
- **Pain:** P3. **Trigger:** board/CTO questioning self-hosting.
- **Evidence:** both sides of one ledger — self-hosted unit cost vs API list pricing, plus latency and queue realities.
- **Output:** per-workload placement economics: current cost ratio vs the API alternative, the **utilization threshold where the answer flips**, and how pending consolidation (S4) moves that threshold. Presented as evidence, not as a routing action — placement changes are change artifacts.
- **Verification:** post-change unit-cost delta settles against prediction.
- **Ritual:** quarterly build-vs-buy review. No competitor can produce this document at all; that is the definition of irreplaceable.
- **Added output — scenario modeling (ratified):** *"What if we consolidate these 3 deployments?"*, *"What if we shift 20% of traffic to the API at the current growth rate?"*, *"What if we add H100s at X utilization?"* Each scenario replays the **actual ledger** under the stated change and prints the resulting placement economics — a replay, not a simulation of invented traffic. Every scenario states its assumptions and is labeled a projection until a real change settles against it. *(Shipped foundation: `hybrid_placement_report` with `flip_utilization`, mixed-estate partitioning, and per-model reconciliation against S2.)*

### S4 — Consolidation Intelligence
- **Pain:** P4. **Trigger:** sprawl discovered during S1/S2 rollout (it always is).
- **Output:** duplicate/overlap deployment map with per-deployment utilization and a consolidation plan as a change artifact (manifest diff + projected unit-cost effect).
- **Verification:** post-merge utilization and unit-cost settlement.

### S5 — Queue Economics & Batch-Window
- **Pain:** P1/P5. **Output:** queue wait translated to business impact; highest-value queue optimizations; off-peak deferral windows (policy: deferral tags in-process; artifact: batch cron). **Verification:** wait-time and throughput deltas.

### S6 — Model Portfolio Intelligence
- **Pain:** P3/P4. **Output:** cost–latency–capacity tradeoff table across the deployed portfolio and API alternatives; portfolio change recommendations as evidence packs. (Quality dimensions enter only where the customer supplies their own eval scores — we never fabricate a quality metric.)

### S7 — SLO Economics
- **Pain:** P5. **Output:** the operational and economic consequence of each latency target; the cost of the current p99 vs relaxed alternatives; recommendation records for SLO revisions (always artifacts — SLOs are the customer's contract, never ours to touch).

### S8 — Procurement Intelligence
- **Pain:** P6. **Trigger:** renewal/expansion on the desk (this trigger has a *date*, which makes it the easiest deal to time).
- **Output:** the **procurement evidence pack** — what the current fleet produced (from the ledger), recoverable capacity not yet recovered (from S1), and the evidence-backed buy/defer/shrink recommendation. Not "buy another GPU" — *here is what the fleet did, here is what recovery yields first, here is the remaining true gap.*
- **Verification:** post-decision capacity and unit-cost tracking against the pack's projections.
- **Commitment renegotiation evidence (P8):** consumed-vs-entitled history across the commitment term, the effective rate that history implies, and whether the next commitment should grow, shrink, or restructure. This is the artifact that turns a true-up meeting from a vendor-led conversation into an evidence-led one.
- **Added — vendor negotiation aids (ratified):**
  - **RFP / renewal artifact templates** carrying projected TCO under different vendor and configuration options, each projection stating its assumptions and labeled a projection.
  - **Spot / instance-type recommendations tied to the customer's own workload profiles** (interruption tolerance derived from declared deferability, not guessed from content).
  - **Opt-in anonymized benchmarking** against aggregated fleet data, *where consented*. **Hard rule:** aggregated, anonymized, opt-in, with a stated minimum cohort size — or it does not ship (§9.11). The negotiating value of "your $/GPU-hour sits in the Nth percentile of comparable fleets" is real, and it is worth exactly nothing if a customer can be identified from it.

### S9 — Energy & Sustainability
- **Pain:** P7 (EU). **Output:** per-workload energy attribution (estimated, labeled, derivation shown) formatted for CSRD input. Compliance as an *output* of the ledger — budget-unlocking in the EU, never the pitch lead.
- **Added depth (ratified):** **water-usage estimates (cooling)** alongside energy, and **Scope 3 hooks** consuming cloud providers' published emissions factors. Both are *estimates with derivations shown and factors cited*, never measurements — a fabricated sustainability number is the fastest way to turn a compliance asset into a liability.

### S10 — Capacity Digital Twin *(later phase)*
- **Output:** what-if evaluation of execution changes by **replaying the ledger** under altered assumptions (different placement, batching, portfolio) — a ledger replay, not an infrastructure simulator. Positioned last because it monetizes accumulated ledger history, which also means it deepens with tenure: the twin is only as good as the ledger is long. Another switching cost.
- **Added — scenario library and risk modeling (ratified):** named, repeatable scenarios — *"consolidate these 3 deployments"*, *"add H100s at X utilization"*, *"shift 20% of traffic to a cheaper model"* — plus **Risk/ROI ranges built from historical variance in the calibration store** (§3.10) — implemented as a **Monte Carlo over the customer's own settled-outcome distribution**, not over invented priors. The range is derived from *this customer's own settled history*, which is why it cannot be copied by a competitor and why it narrows with tenure. Where history is too thin to support a range, the report says so and prints a point estimate with its assumptions — **never a fabricated confidence interval**.

### S11 — Execution Continuity *(candidate to join the S1–S3 spearhead)*
- **Pain:** self-hosted serving degrades and dies — OOMs, saturation, bad deploys — and the customer-facing AI feature goes down with it. **Trigger:** the last incident; reliability budgets are the oldest and fastest-opening budgets in infrastructure.
- **Capability (§3.7):** degradation detection from our own request stream, policy-driven failover among declared endpoints (including declared API fallbacks), automatic return, canary-guarded.
- **Output:** the continuity report per incident — window, requests protected, latency held, cost delta of the failover period — settled in the ledger.
- **Why us:** failover across the *hybrid* boundary (own cluster → API and back) requires both sides in one ledger to price and verify it. Gateways can fail over; only we can prove what it protected and what it cost.

### S12 — Peak Protection & Spend Guards
- **Pain:** P1/P5 congestion at peak; runaway spend from loops and misconfigured jobs (near-universal, and the horror stories are the customer's own).
- **Capability (§3.7):** priority shedding keyed on declared metadata; budget circuit breakers per team/feature.
- **Output:** protected-SLO evidence at each peak event; every guard trip ledger-logged with what was shed/blocked and what it protected.
- **Boundary note:** enforcement primitives ship here; organization-wide policy governance is TokeNet — same ledger, deliberate upsell seam, no duplication.

### S13 — Anomaly & Continuous Improvement *(new; candidate for early paid tier)*
- **Pain:** silent regressions and waste that accumulates unnoticed between reviews. Nobody is watching the ledger daily; by the quarter's end the drift is expensive and its cause is cold.
- **Evidence:** the Anomaly & Regression Detector (§3.5) settling against historical baselines from the same ledger.
- **Output:** a weekly **Execution Health Report** — settled anomalies, recalibrated baselines, and the top five emerging opportunities ranked by value × confidence × time-to-value.
- **Alerting:** ledger-driven alerts (queue wait breaching SLO for revenue traffic; utilization drop signaling idle waste; cost ratio drifting away from the hybrid flip threshold) delivered **into the customer's existing incident tooling (PagerDuty, Opsgenie, Slack)**. We feed their pager; we never become the pager, and we are never in the request path for it.
- **Ritual:** weekly engineering sync **and** monthly exec review. This is the service that makes Tokeymeter *operationally* embedded rather than merely financially embedded — it is why the product is opened weekly instead of monthly.

### S14 — Multi-Cluster / Hybrid Fleet Intelligence *(new; expansion moat)*
- **Pain:** large enterprises run several clusters — on-prem plus one or more clouds, often several regions — with no unified economic view. Cost variance between clusters is invisible, and policy drift between them is discovered only after an incident.
- **Evidence:** one ledger spanning every cluster and the API, joined on the endpoint identity and cost model already shipped.
- **Output:** unified fleet economics; per-cluster unit-cost variance with derivation; **inconsistent-policy detection** across clusters; placement recommendations that account for the whole estate rather than one cluster in isolation.
- **Ritual:** quarterly infrastructure strategy review; annual capacity planning.
- **Why it is an expansion moat:** the value scales with the number of clusters, and it is the natural land-and-expand path from a single-cluster pilot to an estate-wide contract. It also raises the switching cost superlinearly — replacing Tokeymeter means rebuilding a *unified* history, not one cluster's history.

### S15 — Governance & Policy Layer *(the TokeNet upsell bridge)*
- **Pain:** in a large org, policies are authored ad hoc, applied inconsistently across teams and clusters, and audited never. When something goes wrong nobody can say who approved what, when.
- **Output:** centralized **policy authoring and auditing in the same ledger** — versioned policies, approval workflow, full change history, and per-policy settlement evidence. Enforcement remains exactly where §4 puts it: demand-side in-process, or emitted as a change artifact.
- **Boundary note (important):** this is the deliberate **seam to TokeNet**, not a duplicate of it. Tokeymeter holds the *policy record and its economic evidence*; TokeNet owns organization-wide identity, governance, and the command plane. Same ledger, clean upsell, no fork — the standing anti-fork decision (§9.6) applies here in full.
- **Multi-tenancy & RBAC:** team-scoped views with central oversight — a team lead sees their own chargeback and capacity; the platform owner sees the estate. Required for large-org rollout and for the security review that precedes it.

---

## §6 — Irreplaceable and Unskippable (the moat mechanics)

**Irreplaceable** — five mechanisms, all structural:
1. **The hybrid ledger.** Only system holding API + self-hosted execution in one schema, one pricing frame. Every S3 document is a proof of uniqueness.
2. **The calibration flywheel.** Estimator accuracy is per-customer state built from settled recommendations. A competitor starts at zero accuracy *and has to admit it*.
3. **Ledger continuity in finance's books.** Once S2 runs the monthly close, the ledger is load-bearing for chargeback. Removal breaks a business process, not a dashboard.
4. **Verified change history.** Every optimization ever applied, with its settled outcome, is institutional memory. Leaving Tokeymeter means abandoning the organization's own execution history.
5. **Ritual & export lock-in.** Outputs become the *source of truth* for the finance close, procurement decks, and compliance filings. Once numbers with our provenance are in filed documents, replacement costs a data migration **plus a re-validation of history** — and someone must sign that the new numbers still tie to the old filings. That signature is the switching cost.

**Unskippable** — the design rule: **every service output is the named input to a recurring ritual.** Monthly cost close (S2). Monthly capacity review (S1). Quarterly build-vs-buy (S3). Procurement events (S8). Annual CSRD filing (S9). Change-management PRs (artifact emitter — Tokeymeter artifacts appear in the customer's own review queue). Products bolted to rituals don't get skipped when a budget review comes; they *run* the budget review.

**Unskippability rule (ratified, enforced at phase exit):** *every paid service must produce at least one artifact that appears in a customer ritual within the first 30 days of activation.* A service that cannot name its ritual and hit that window does not ship as paid — it is either merged into one that can, or killed under §9.

---

## §7 — Packaging & Revenue

**OSS, free forever (the wedge, already shipped):** in-process sensor, execution ledger, doctor, demo, secret firewall, PII redaction, basic savings & capacity reports. The OSS tier must remain genuinely excellent — it is distribution, trust, and the instrumentation beachhead.

**OSS leverage rule (ratified):** the free tier must include **enough ledger and enough basic reporting to genuinely hook a user** — they should be able to see their own execution truth and want more, not hit a wall at first contact. The paid unlock path is *structural and legible*: telemetry-ingestion adapters, the declared cost model, the full optimization + verification engine, and the action surface. **The wedge is not crippled software; it is a complete small thing that makes the large thing obviously worth buying.** A user who never connects telemetry should still be glad they installed it.

**Paid — Execution Intelligence Services:** telemetry-ingestion adapters, declared cost model, unit-cost engine, optimization engine + recommendation records, verification harness + calibration, change-artifact emitter, S1–S10. Gating follows the already-set structure: **telemetry-gated, pilot-unlocked** — the paid layer activates when the customer connects telemetry and enters a pilot.

**Pilot structure (the revenue motion):**
- Paid pilot, fixed length (target 4–6 weeks), on the customer's real cluster.
- Weeks 1–2: pain validation (§10) — the customer's own ledger proves or disproves P1–P7. Disproven pains are dropped from scope *in writing*.
- Success criteria set jointly at kickoff, in the only currency we deal in: **N recommendations settled-verified** (N and the metrics chosen with the customer — never asserted by us).
- Pilot converts to an annual subscription on settlement evidence, not on a demo.

**Pricing structures to test (structures, not numbers — numbers are set per pilot and never invented in this document):**
1. Per-GPU-node per month (simple, scales with estate, procurement-legible)
2. Per-cluster flat tier (predictable, favored by finance)
3. Gain-share on settled-verified recovery — *pilot phase only*, converting to flat at renewal (aligns incentives early, avoids perverse long-term incentives and audit friction later)
4. Platform fee + per-service tiers (S1–S3 core; S8/S9 as event/compliance add-ons)

**Pricing decision rule:** price against the *ritual* each service feeds (a monthly close, a procurement event, a compliance filing), never against tokens — token-metered pricing would contradict our own optimization incentive and reintroduce the perverse economics we exist to fix.

**Paid-tier positioning (say it this way):** we are **not selling "optimization software."** We are selling **provable execution advantage and reduced risk on multi-million-dollar AI infrastructure decisions.** Price against the value delivered to rituals — capacity recovered, chargeback enabled, procurement de-risked, compliance simplified — never against a feature list.

**Tiering (ratified):**
- **Core paid layer — S1–S3.** Capacity Recovery, Unit-Cost & Chargeback, Hybrid Placement. This is the spearhead: it opens the door, locks it, and makes the product irreplaceable. *(All three shipped.)*
- **Add-ons / higher tier:** S8 Procurement, S9 Energy, S11 Continuity, S13 Anomaly.
- **Enterprise tier:** S14 Multi-Cluster, advanced scenario replay, dedicated support.

**Pricing anchors.** Per-GPU as the base unit, plus value elements (gain-share on verified recovery, **pilot only**). Sell against the **cost of inaction** — continued waste, audit exposure, and procurement decisions made without evidence — never against a feature list.

**Pilot success criteria (strengthened).** At minimum: **2–3 settled-verified recommendations** *and* **at least one artifact that landed in a real customer ritual**. Plus an **estimator accuracy baseline recorded at pilot end** — the first point on the calibration curve, and the number that proves the flywheel is real at renewal.

**Expansion path:** self-hosted paid layer → **TokeNet** (identity, policy, governance) → **Enterprise Suite** (multi-cluster, advanced scenario replay, dedicated support) — all on the *same* ledger. The ledger is the bridge; nothing in this document forks it.

### Messaging framework (sales guide — use verbatim by audience)

- **Platform / Engineering:** *"Close the loop on every optimization with verification. Safe demand-side actions. No more guesswork."*
- **Finance / FinOps:** *"Unit costs on your hardware. True chargeback. Auditable numbers with provenance."*
- **Leadership / CTO:** *"Data-driven build-vs-buy. Procurement evidence that stands up to scrutiny. Hybrid economics in one view."*
- **All audiences:** *"We make self-hosted AI economically defensible and continuously improving — with proof."*

**Objection handler — "another dashboard?"** Lead with the four things a dashboard structurally cannot do: it does not hold both sides of the estate in one priced ledger; it does not attach a settlement query to its own advice; it does not act on the demand side; and it does not survive an audit of where its numbers came from. Then show a settled recommendation — including a *failed* one.

---

## §8 — Build Sequence (test-gated phases; no phase ships with estimates presented as measurements)

### Phase S0 — Self-hosted instrumentation *(foundation)*
Ledger extensions (§3.1) including the **tokens_provenance fix for the shadow-hit estimated-tokens bug**; endpoint identity; queue_wait capture from vLLM/TGI where reported. Wrapper verification against vLLM, TGI, Ollama via their OpenAI-compatible endpoints (existing wrappers; adapters only where reality differs). All Windows Week-1 ledger items land here: cp1252-safe CLI output, encoding-explicit open() tree-wide, timezone-independent keys, relative (not absolute) perf gates.
**Exit:** `tokeymeter demo` runs against a local Ollama/vLLM with an honest ledger; full suite green; provenance labels correct on every record.

### Phase S1 — Unit cost + flagship report *(first sellable artifact)*
`register_cluster_costs` (§3.3); unit-cost engine (§3.4); **Capacity Recovery Report v1** (evidence-only; no actions yet); honesty block throughout; side-by-side API-list comparison column.
**Exit:** report generated from a real pilot cluster's data with every number carrying provenance; this artifact alone must be worth paying for.
**Priority note (ratified):** the Capacity Recovery Report **plus** the unit-cost side-by-side **plus** the honesty blocks are, together, the minimum sellable unit. Ship all three or none.

### Phase S1.5 — Export & ritual integration layer *(parallel with S1; ratified addition)*
The §3.12 layer: finance CSV/GL export, procurement evidence pack, webhook/OTel ritual hooks, audit-log export, and the first pre-built adapters (K8s + vLLM/TGI; cloud lease/invoice import for the Declared Cost Model).
**Why parallel, not later:** an artifact that cannot reach the customer's existing process arrives late to the ritual it was built for, and the 30-day unskippability rule (§6) fails on integration friction rather than on value. Value that cannot be delivered into a meeting is not yet value.
**Exit:** the S1 report reaches a real customer ritual through an export or hook — not a screenshot pasted into a deck by us.

### Phase S2 — Optimization engine + verification *(the loop closes)*
Recommendation record schema (§3.6); launch detectors (idle-window, consolidation, cache-opportunity, deferral, hybrid-threshold); ROI estimator with floors; verification harness (§3.9).
**Exit:** at least one recommendation settled-verified end-to-end on a pilot cluster — the loop demonstrated on real hardware, including at least one settled-*failed* handled and displayed correctly.

### Phase S3 — Bounded action *(the "active" edge, safely)*
In-process policy actuator (§3.7) with canary rollout, auto-revert, and fail-open verified under kill tests; Execution Continuity (S11) and budget guards (S12) as the first two revenue-bearing policies; change artifact emitter (§3.8) for consolidation and batch-window artifacts; advisory signals (§3.11).
**Exit:** one continuity failover exercised and settled on real traffic; one budget guard trip correctly logged; one change artifact merged by a customer through their own pipeline and settled; fail-open demonstrated (kill Tokeymeter mid-policy, traffic flows default).

### Phase S4 — Portfolio depth
S6/S7/S8/S9 services; calibration store + estimator accuracy report; S10 twin (ledger replay) last.
**Exit per service:** its output feeding a real customer ritual at least once.

Standing gates for every phase: full test suite green (engine + runner + stress); cross-platform (Linux + Windows east-of-UTC); every new number provenance-tagged; every new feature answers the §9 checklist.

**Cross-phase demo gate (ratified):** every phase exit demo must show its output **feeding a simulated customer ritual** — a generated monthly close packet, a procurement pack, a review agenda — not a terminal dump. If the output cannot be shown landing in a meeting, the phase has not exited.

---

## §9 — Failure Modes (kill list — check every feature against it)

1. **Dashboard drift.** A view with no recommendation record behind it is decoration. Every screen must terminate in a recommendation, a settlement, or a ritual artifact.
2. **Orchestration creep.** One question per feature: *does it mutate infrastructure state?* If yes, it is a change artifact or it is cut. No exceptions for "small" mutations.
3. **Collector creep.** The moment we ship an exporter or a node agent, we are a worse Prometheus with better marketing. Ingestion only.
4. **ML theater.** No "learns," "AI-powered," or "intelligent" claims ahead of shipped mechanism. Deterministic detectors + calibration is the honest story and it is a *stronger* enterprise story.
5. **Fabricated value.** No projected-savings number without assumptions inline and a floor that settlement can fail.
6. **The TokenHost fork.** Any proposal to split self-hosted into a separate **product with its own ledger** is rejected by standing decision. The line, stated precisely so this question stops recurring:

   > **The product surface may split. The data plane may never split.**

   **Allowed (and encouraged where it helps the sale):** a distinct name, landing page, docs set, pricing SKU, deployment profile, or sales motion for self-hosters — up to and including calling that surface "TokenHost" externally. Go-to-market can be as separate as the buyer requires.

   **Vetoed, permanently:** a second ledger, a second record schema, a second pricing/rounding discipline, a second codebase, or **any design where hybrid analysis requires joining two systems.**

   **The test for any future proposal:** *does a hybrid answer require reconciling two stores?* If yes, it is the fork wearing a new name — reject it.

   **Why the veto is architectural, not stylistic** — the empirical case from the S2/S3 build:
   - The moat *is* the single ledger (§0 moat #1). A hybrid answer that requires joining two systems is precisely the problem the customer already has (Prometheus for the cluster, provider dashboard for the API, nothing that joins them). Shipping two products plus a connector recreates their problem with our logo on it.
   - **Exact cross-report reconciliation was only achievable inside one codebase.** S2 and S3 reconcile per-model exactly because they import the *same* sanitizer, share a record schema, and apply an identical rounding discipline — literally the same function object making identical exclusion decisions on malformed records. Across two products those become a **versioned integration contract**, and every skew between release cycles becomes a "the numbers don't match" bug in front of a CFO.
   - The mixed-estate defect found in the S3 re-audit is the proof: a real ledger holds self-hosted **and** API-served records together, and correctness required the *operator declaring which models are self-hosted* — a one-line parameter within one ledger. Across a product boundary it becomes a distributed-consistency problem for a number finance has to sign.
   - Reconciliation cost compounds: totals across two independently-rounded reports already drift sub-micro-dollar by rounding order *within* one codebase (an accepted, documented limit). Across two products with independent release cycles, that drift becomes unbounded and undebuggable.
7. **Content-aware creep.** Any feature requiring prompt/response content is out of scope for this product line, full stop.
8. **Gateway creep.** Any design that requires traffic to route *through* a Tokeymeter service is rejected; we live in-process or we read the ledger.
9. **Alert-tool creep.** S13 alerts are delivered *into* the customer's existing incident tooling. The moment we build an on-call rotation, an escalation policy, or an incident timeline UI, we are a worse PagerDuty. We emit; they page.
10. **Simulation theater.** Scenario modeling (S3/S10) replays the **real ledger** under a stated change. Any "simulation" of invented traffic is fabricated value under (5) wearing a nicer name.
11. **Benchmark-leak creep.** Cross-customer benchmarking (S8) ships only as opt-in, aggregated, and anonymized — or it does not ship. One identifiable leak ends the company's credibility, and the feature is worth less than the trust it would cost.

---

## §10 — Pilot Discovery Instrument (weeks 1–2 of every pilot)

For each pain: one discovery question at kickoff + one ledger/telemetry test that proves or disproves it with the customer's own data.

| Pain | Kickoff question | Data test (weeks 1–2) |
|---|---|---|
| P1 capacity | "When did you last decline a workload for capacity, and what sat idle that night?" | Occupancy + queue-wait profile by hour; idle-window enumeration |
| P2 unit cost | "What does one request cost on your cluster? Who pays for the shared one?" | Attempt the unit-cost table; if inputs exist, print it in week 1 |
| P3 build-vs-buy | "At what utilization does self-hosting model X beat the API? Who last computed that?" | Hybrid comparison on live traffic |
| P4 sprawl | "How many deployments of the same base model exist across teams?" | Endpoint-identity dedup scan |
| P5 SLO/queue | "Who set the current latency target, and when was it last revisited?" | Queue-wait cost profile; SLO sensitivity sketch |
| P6 procurement | "What evidence went into the last GPU purchase?" | Fleet-output summary from available history |
| P7 energy | "Can you attribute AI energy per workload for CSRD today?" | Estimated attribution with derivation shown |

Gate at end of week 2: pains that validated define pilot scope; pains that didn't are dropped **in writing**. Honesty as a sales weapon: the vendor that narrows its own scope on evidence is the vendor whose remaining claims get believed.

---

## §11 — Success Metrics & Customer Outcomes *(new)*

### Definition of Done — customer view

The minimum bar for customer success and the North Star for the whole product line. A self-hosting customer can:

1. **See every AI request in one unified, content-blind ledger** across API and self-hosted hardware. *(SHIPPED)*
2. **Read true unit cost with full provenance** — $/request and $/1K tokens on their own cluster, side-by-side with the API in the same units. *(SHIPPED — S1/S2)*
3. **Receive ranked, evidence-backed recommendations**, each with a predicted floor and a pre-attached settlement query. *(SPEC — Phase S2)*
4. **Approve bounded demand-side policies** that are canary-guarded, auto-revert, and fail-open. *(primitives PARTIAL; policy set SPEC — Phase S3)*
5. **Merge infrastructure changes as artifacts** through their own pipeline, with automatic verification after merge. *(SPEC — Phase S3)*
6. **Open a monthly report** showing verified wins, failed settlements, and improving estimator accuracy — finance-auditable proof. *(close packet SHIPPED; verification/calibration SPEC — Phase S2)*

**Cost savings are the natural byproduct of better measured, verified, and improved execution — never the marketed identity (§0).** The status marks are deliberate: this list is the finished product, and today the measurement and ledger half is shipped while the verification and action half is the remaining build. The bar does not move; our position against it is stated honestly.

### Outcome targets (sales use — discipline required)

Target outcomes such as *substantial capacity recovery within the first two quarters, chargeback enabled, renewals de-risked, and CSRD-ready reporting* are **positioning claims, not product promises**. The rule that governs every one of them:

> **No outcome percentage is ever quoted to a customer as our number.** Market anchors are cited as external `[market claim]` with a source. The only figures we present as ours are (a) the customer's own baseline from the week-1–2 pilot instrument (§10), and (b) settled-verified deltas from their own ledger. A settled number from their cluster beats any industry statistic in the room — and unlike the statistic, it cannot be argued with.

This is not conservatism; it is the sales weapon. Every competitor arrives with a percentage from a whitepaper. We arrive with the customer's own ledger and a settlement query they can run themselves.

### Executive artifacts (the recurring proof)

- **Quarterly Estimator Accuracy Report** — how close our predictions came, including failures, per detector class. This is the artifact that makes the calibration flywheel *visible* to the buyer, and it is deliberately uncomfortable: showing failed settlements is what makes the verified ones credible.
- **Annual Verified Value Statement** — every settled-verified recommendation for the year with its measured delta, suitable for a budget defense.

### Enterprise readiness track (runs alongside services)

SOC 2 / ISO readiness; the §4 boundary table as a **standing security-review artifact** with the fail-open and no-mutation guarantees demonstrated, not just asserted; multi-tenancy and RBAC (S15); data-residency posture for the ledger. **This track gates enterprise deals independently of feature work** — a service that cannot pass security review cannot be sold, regardless of how good the artifact is.

---

## §12 — Why Companies Will Definitely Pay *(the four-line answer, for any room)*

1. **Real utility.** It solves validated, expensive pains with measurable, auditable wins — not "insights." Every number carries provenance; every recommendation carries a settlement query the customer can run themselves.
2. **Unskippable.** It embeds into finance, procurement, engineering, and compliance rituals. Removal does not degrade a dashboard — it **breaks a process** that someone has to sign.
3. **Irreplaceable.** The hybrid ledger, the calibration history, and the verified change log create switching costs that compound with tenure. No other tool holds both API and self-hosted execution in one auditable book, and a competitor arriving on day one starts with **zero** history and has to say so.
4. **Risk mitigation.** Honesty is the sales instrument: failed settlements are displayed, the action boundary is printed, and every number states whether it was measured, declared, or estimated. This is what removes friction from enterprise security and finance review — the two places where deals actually die.

**The one-line version:** *the alternative to buying this is continued waste, blind procurement decisions, and audit exposure — in an environment where AI spend is under more scrutiny every quarter.*

---

## Definition of Done (for this product line, at any moment)

A self-hosting customer can: see every AI request in one content-blind ledger across API and owned hardware; read a unit cost with shown derivation; receive ranked, evidence-backed recommendations each carrying a predicted floor and a pre-attached settlement query; approve bounded in-process policies that auto-revert on undershoot; merge infrastructure changes as artifacts through their own pipeline; and open a monthly report where verified wins, failed settlements, and estimator accuracy appear side by side — a ledger finance audits, a loop engineering trusts, and a set of documents no other vendor can print.
