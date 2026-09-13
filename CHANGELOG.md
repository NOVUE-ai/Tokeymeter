# Changelog

## 0.31.1 — 2026-09-07

First public release.

- Repository slug set to `NOVUE-ai/tokeymeter`; every README link and image is
  an absolute URL so the same text renders on GitHub and as the PyPI long
  description.
- README rebuilt around a real run: 98 requests reconciling exactly with
  OpenAI's usage page, 87% of the spend going to tasks that produced nothing.
- `agents --html` redesigned around a progress spectrum, with the NOVUE mark,
  frosted panels and CSS-only interaction. No script, no network request.
- `watch` enables ANSI on Windows and sizes each frame to the terminal, so the
  live view replaces itself instead of scrolling.
- Internal architecture notes, red-team findings and stale build manifests
  removed from the distributed tree.
- `integrations/tokenet` (the control-plane client) removed: it is the paid
  surface, not part of the open-source distribution. The checks and tests that
  exercise it now skip when it is absent, so the gate passes on the code that
  actually ships.
- `MANIFEST.in` added: the sdist previously carried `tests/` without the
  `docs/` and `examples/` those tests read, so a downloaded source tree failed
  11 of its own tests.

## [0.15.0] — SELF-HOSTED ENTERPRISE BUILD (S0 → S1.5-2)

The self-hosted offering: one ledger spanning API and owned hardware, with
three board-grade artifacts that reconcile against each other, plus the export
layer that carries them into a customer's monthly close.

Build spec: `TOKEYMETER_SELF_HOSTED_BUILD_DOC.md` (§8 build sequence).

### Phase S0 — self-hosted instrumentation (the join keys)

- **S0-1 shadow-hit token provenance.** Shadow-hit records are stamped with the
  real call's provider-reported token counts instead of a `chars/4` estimate.
  Covers sync/async/stream plus exception, hard-cap, and abandoned-stream paths
  via `_shadow_flush_fallback`, which also drains the reported-usage contextvar
  (closing a latent stale-bleed). Shadow mode is the pre-sales evaluation mode,
  so "would have saved" now rests on provider truth.
- **S0-2 `endpoint_identity` + `queue_wait_ms` on `CallRecord`.** New
  `engines/execution/endpoint.py` mirrors `identity.principal` (contextvar +
  `endpoint()` context manager; precedence explicit > contextvar > None).
  Content-blind: URL-shaped values are rejected at decoration. `queue_wait_ms`
  is miss-only, consume-once, absent-reads-None-never-0; schema-tolerant
  extraction in all three SDK wrappers, never fabricated from client timing.
- **S0-3 record-construction parity.** Shared `savings.build_call_record`
  factory; closed a real gap where the async OpenAI wrapper silently omitted
  `pricing_source` / `principal` / `key_name`.

### Phase S1 — unit cost and the first sellable artifact

- **`register_cluster_costs`** — capital, power, facility, and staff components
  each derived and shown separately; owned XOR leased, one required.
- **Capacity Recovery Report** (`engines/economics/capacity_report.py`) —
  recovered capacity by MEASURED mechanism only; roadmap mechanisms appear
  under `not_yet_measured` carrying ZERO numbers; USD appears only when a rate
  is supplied. Mechanism rows sum exactly to the total (largest-remainder
  apportionment, chosen precisely so the column adds up).
- **S1.1 true token recovery** — a cache hit recovers the original miss's true
  token counts from the envelope instead of estimating from the cached value.
  Cache envelope extended backward-compatibly with a third element (token
  meta) plus a `meta()` accessor.

### Phase S2 — Unit-Cost & Chargeback Ledger

- **`chargeback_report` / `chargeback_csv`** (`engines/economics/chargeback.py`).
  Spend by tag / model / endpoint_identity / principal / key_name over half-open
  `[start, end)` epoch periods.
  - Reconciliation by construction: every total is the sum of the displayed
    rounded rows.
  - SPEND = EXECUTED. Cache hits are an `avoided_*` credit and are never netted
    into spend; shadow REAL calls are spend; shadow HITS are excluded and
    counted.
  - No record is lost: unattributable rows land in an `(unattributed)` bucket.
  - Per-row provenance; fallback pricing flagged; CSV formula-injection defense.

### Phase S3 — Hybrid Placement Intelligence

- **`hybrid_placement_report` / `hybrid_placement_csv`**
  (`engines/economics/hybrid.py`). Booked self-host cost vs API-equivalent cost
  for the SAME executed volume, priced at each row's OWN token mix, with the
  ratio, the verdict, and `flip_utilization` — the utilization at which the
  build-vs-buy answer flips.
  - Executed volume uses EXACTLY the S2 semantics, so the two board artifacts
    reconcile per-model.
  - Registry default/fallback prices are treated as UNPRICED and excluded from
    verdicts — a board decision never rests on a generic guess.
  - `utilization > 1.0` is reported AS IS with a flag, never clamped.
  - `selfhosted_models` partitions a mixed estate: API-served workloads are
    summarized separately and never counted as self-host cost.

### Phase S1.5 — export and ritual integration

- **S1.5-1 Close Packet + General Ledger export**
  (`engines/economics/close_packet.py`). The packet is an AUDIT, not a bundle:
  it composes the shipped reports and proves they reconcile before finance sees
  them, emitting `reconciliation.status` = `reconciled` | `DISCREPANCY` with
  every check listed, and a separate `warnings` channel for caveats that are
  not disagreements.
  - GL account codes are OPERATOR-DECLARED, never guessed; unmapped cost
    centres post to a declared suspense account and are listed in
    `unmapped_cost_centers`. Money is never lost by that routing.
  - Reconciliation tolerance scales as `max(1e-4, 5e-6 * sqrt(rows))` because
    display-rounding error is a random walk; a fixed tolerance false-alarms on
    large estates and a linear one is loose enough to hide real errors.
- **S1.5-2 period-bounded capacity.** `capacity_recovery_report` accepts
  `period_start` / `period_end` / `records` and rebuilds its aggregate through
  the SAME sanitizer S2/S3 use, so all three artifacts make byte-identical
  exclusion decisions. A record is recovered capacity iff `hit AND NOT shadow`.
  The close packet's capacity scope WARNING became three real CHECKS
  (`capacity_period_matches`, `capacity_exclusions_match_chargeback`,
  `capacity_recovered_matches_chargeback_avoided`).

### Hardening passes (each found real defects in shipped code)

- **Stress:** mechanism rows failed to sum under mixed mechanisms (banker's
  rounding) → largest-remainder apportionment; NaN `tps` and NaN
  `gpu_hour_rate` slipped past `<= 0` / `< 0` guards (NaN compares False) →
  `isfinite` checks.
- **Real-circumstance:** a writer killed mid-write left a torn tail with no
  newline and the next process's append concatenated onto it, silently LOSING
  one record per crash → `_ensure_clean_tail_locked`. First over-the-wire
  integration test (real OpenAI SDK → local vLLM-schema server → real wrapper →
  ledger) confirmed the SDK preserves vLLM's `usage.queue_time_ms` via pydantic
  `model_extra`.
- **S1 re-audit:** displayed cost components were rounded to 6dp but the stated
  `gpu_hour_rate` came from unrounded values, so the displayed column did not
  sum to the stated total → stated rate is now the sum of displayed components,
  with `gpu_hour_rate_usd_full_precision` retained for downstream math.
- **S3 re-audit:** a mixed estate mislabeled API spend as GPU spend →
  `selfhosted_models` partition, `api_served_workloads`, `estate_mode`.
- **Hostile data:** corrupt ledger records (bit-flips, partial writes that
  still parse) either crashed the report or silently poisoned totals — NaN and
  inf propagated, and a NEGATIVE cost silently REDUCED the bill. Shared
  sanitizer (`_record_numbers` / `_record_ts`) now excludes corrupt records
  WHOLE and counts them in `excluded_malformed_records`; dimension values are
  string-coerced so a numeric id is retained rather than crashing `sorted()`.
- **S1.5-1 deep re-audit:** seven defects fixed, including a tolerance that
  false-alarmed on a healthy 50k-row close, a packet that was not a snapshot
  (mutating a source report changed the filed packet), NaN totals passing checks
  vacuously, and an `api_served` total of 999 against rows summing to 5 that
  passed as "reconciled".

### Fixed — integrity self-check was silently inert (pre-existing)

`tokeymeter/engines/trust/integrity.py` defaulted `package_dir` to
`dirname(__file__)`, which resolves to `tokeymeter/engines/trust/` — not the
package root where `_manifest.json` ships. The documented zero-argument call,
`verify_self()`, therefore returned `no_manifest` on every installed wheel,
silently disabling tamper detection. Added `_default_package_dir()`, which
resolves the real package root, and repointed all three call sites
(`generate_manifest`, `write_manifest`, `verify_self`).

The shipped manifest is also now regenerated at package time (109 → 131 files
hashed). It had gone stale: the package reported *itself* as `tampered`, which
is worse than having no manifest — it cries wolf in exactly the security review
it exists to satisfy. Six regression tests pin the resolver, prove the check
actually fails on a modified file, pin the honesty posture that an unsigned
manifest reports `unverified` rather than quietly downgrading to `ok`, and
assert that every in-package file is declared shippable.

Also fixed: `[tool.setuptools.package-data]` did not ship the two `.md` files
that live inside the package, so an installed wheel reported itself `tampered`
for files that were simply never installed. Found by installing into a clean
virtualenv and verifying the wheel rather than the source tree.

### Fixed — cross-platform portability pass

- **`tokeymeter demo` crashed on legacy consoles.** A `\u2192` in its output
  raised `UnicodeEncodeError` on cp1252 (the Windows default in many
  environments), killing the command. Added a console guard in the CLI entry
  point that sets `errors="replace"` on stdout/stderr — keeping the stream's own
  encoding, since rewriting to UTF-8 would emit bytes a cp1252 console renders
  as mojibake — and replaced the non-ASCII glyphs in CLI output with ASCII so
  legacy terminals show readable text rather than `?`. All subcommands
  (`version`, `doctor`, `demo`, `pricing`, `report`) verified under
  `PYTHONIOENCODING=cp1252`.
- **The README gate crashed on Windows runners.** It read `README.md` via
  `read_text()` with no encoding, inheriting cp1252 — and README.md contains
  bytes cp1252 cannot decode. Now explicit UTF-8.
- **13 text-mode `open()` calls in shipped code had no encoding**, including
  the savings ledger, the integrity manifest, audit proof export, and warmup.
  All now explicit UTF-8. (The ledger itself was already protected in practice
  by `json.dumps` ASCII-escaping, but reads of externally-written files were
  not.)
- **Local-state paths ignored `TOKEYMETER_HOME`.** The audit DB, install
  secret, audit signing key, cache DB, semantic index, and memory DB all
  hardcoded `~/.tokeymeter/<name>`, bypassing both the env var and
  `set_home()` — breaking containers, CI, and locked-down service accounts
  where `~` is unwritable or shared. Added `paths.state_path()` and switched
  all six to lazy resolution (a `None` default resolved at construction, since
  the home may be configured after import). An explicit path still wins.

`tests/test_platform_portability.py` (15 tests) pins all of it, including
guards that fail on any future hardcoded home path or encoding-less text open.

### Measured — Windows ledger write performance

A 60,000-record durability test that takes 1.9s on Linux takes **95.9s on
Windows**. Root cause identified and quantified rather than guessed: the
synchronous ledger performs 60,552 individual file opens (one per record), and
Windows charges ~1.5ms per open/close against Linux's ~25us once NTFS metadata
and real-time AV scanning are counted. 60,552 x 1.5ms is ~91s, matching the
95.9s observed.

**Not a defect.** Re-opening per append is what keeps the ledger correct under
concurrency: another process may trim and `os.replace` the file at any moment,
and a cached handle would then append to a dead inode. Nor is it a production
constraint — one record equals one model call, so even the Windows figure of
~626 records/second far exceeds what a single process generates.

Buffered mode (`set_buffered_savings(True, buffer_size=128)`) cuts file opens
~65x for genuinely high-volume or slow-disk deployments, with the trade-off
stated explicitly: unflushed records are lost on an abrupt kill. Documented with
the full measurement table in `docs/self-hosted/WINDOWS_PERFORMANCE.md`, along
with the Windows pytest isolation flags (`--basetemp`, `-p no:cacheprovider`)
that avoid shared-temp access-denied errors.

File-open-heavy tests are now marked `io_heavy` so local iteration can deselect
them (`-m "not io_heavy"`). No test was weakened to make it faster.

### Testing

Suite grows to **1273 passing** (from 1024 at v0.14.0); repo gate
`scripts/run_all_checks.py` 17/17. New suites: `test_shadow_reported_provenance`,
`test_selfhost_ledger_fields`, `test_record_parity`, `test_capacity_recovery`,
`test_s1_1_token_recovery`, `test_chargeback`, `test_hybrid_placement`,
`test_close_packet`, `test_ledger_crash_consistency`,
`test_wire_integration_openai`.

Verification beyond unit tests, run against every artifact: multi-process
ledgers, kill-during-write crash consistency, hostile/corrupt records,
mixed-estate partitioning, 200k-record scale replays, and cross-artifact
reconciliation fuzz (per-model exactness proven over hundreds of trials).

### Known limitations (stated, not hidden)

- In-memory async single-flight followers receive a raw value rather than an
  envelope, so they estimate recovered tokens instead of recovering them. The
  direction is safe (understates), and the distributed single-flight path does
  recover. One path only.
- The async OpenAI wrapper's `_emit_record` remains a separate path from the
  decorator — a drift source flagged since S0-3.
- Total-vs-total equality across two independently-rounded reports differs by
  sub-micro-dollar rounding ORDER. Per-row and per-model exactness is the
  guarantee; the close packet's tolerance is scaled accordingly.
- Absolute performance gates, a Windows pass, and a smoke test against real
  Ollama/vLLM require the operator's own machine and are not run in CI here.

## [Unreleased] — W0-W9 FULL CERTIFICATION: deep red-team + high-workload stress (+34 tests)

### Added — the hardest categories a hostile enterprise audit runs
Two new suites push the ASSEMBLED W0-W9 runtime to its limits: adversarial
depth on the new platform/universality surface, and extreme workload.

- `tests/test_runtime_redteam_platform.py` (27): deep adversarial red-team on
  W8/W9, organized by real attack class.
  - JWT ATTACKS (12): alg=none injection, signature stripping, forged
    signature, wrong-secret, alg-confusion (EdDSA token -> HS256 verifier),
    expired-by-one-second, nbf-in-future, issuer spoofing, audience confusion,
    malformed-tokens-never-crash, empty/missing principal, claim-injection
    (is_admin/injected_role ignored — only configured claims bind).
  - PLUGIN FORGERY (7): observe->enforce privilege escalation refused, hook
    swap refused, order tamper refused, key-substitution attack refused,
    valid-sig-from-untrusted-key refused, empty/malformed signatures refused,
    hash-field tamper refused.
  - FRAMEWORK INJECTION (3): a governed framework still screens secrets;
    governed_tool screens BEFORE the tool body runs (secret never reaches it);
    framework wrap is content-blind in the trace.
  - CATALOG ISOLATION (3): independent adapters, immutable frozen specs,
    every catalog adapter stays conformant across repeated instantiation.
  - HOTRELOAD RACE (2): 32 readers + flooding writer -> zero torn reads;
    interleaved valid/invalid reloads never leave a rejected config live.
- `tests/test_runtime_stress_workload.py` (7): high-workload, full stack
  (proof+security+optimization+economics).
  - 2000 concurrent full-stack requests: all unique, zero errors, chain valid,
    throughput held.
  - Mixed adversarial traffic: clean + secret + 500KB-giant + unicode-garbage
    interleaved and shuffled — 150/150 secrets blocked, zero crashes, chain
    valid post-chaos.
  - 50k requests memory-bounded (retention cap holds, RSS growth < 15MB).
  - Rate limit EXACT (100) at 1000 threads.
  - Proof-chain integrity under 500 concurrent seals + 20-sample offline
    verify.
  - Budget no-overspend under 500 concurrent spenders.
  - No thread/fd leak over 10k requests.

### Findings — ZERO vulnerabilities, ZERO regressions
Both suites passed clean on the first run. The W8/W9 security surface (OIDC
identity, signed plugins, framework boundaries, multi-provider catalog) resists
every attack in the threat model, and the full stack holds at extreme load
without tearing, leaking, or overspending. No code changes required.

### Verification
- +34 tests; 317 runtime tests green together; runtime mypy clean (25 files).
- CERTIFYING GATE x2: engine 1058/0 · plane 13/13 · runner 18/18 · GREEN both.


## [Unreleased] — W9 COMPLETE + EXIT GATE: universality, conformance, frameworks, tools, packs, hardening

### Added — "any model, any framework, any environment" as engineering fact
- `runtime/catalog.py`: the model catalog. TRUE universality through one
  dialect — 16 providers, the OpenAI-compatible ones reached by a single
  adapter + base_url (Gemini, Together, Groq, Mistral, DeepSeek, Fireworks,
  Perplexity, OpenRouter, xAI, vLLM, Ollama, LM Studio, Bedrock, Azure), plus
  Anthropic's native dialect. `adapter_for(name, client)` / `async_adapter_for`
  build the correct SHIPPED adapter around a caller-constructed client — NOVUE
  never holds credentials. A model that does not exist yet is supported the
  day it ships by pointing the OpenAI adapter at its endpoint. REACH ONLY —
  no new inference path.
- `runtime/conformance.py`: the conformance kit — "works with NOVUE" as a
  checkable claim. `check_adapter(factory)` runs a 7-check battery (get_info,
  health, kernel inference, streaming, usage-truth, error taxonomy,
  content-blind) against any adapter, offline, via a controllable fake client.
  A green ConformanceReport is what "Tokeymeter Verified" means; a broken
  adapter FAILS the kit (the gate has teeth).
- `runtime/frameworks.py`: any framework, governed the same way. `wrap_callable`
  is the universal guarantee (ANY prompt->response callable becomes a governed
  Runtime); `wrap_langchain_llm` (.invoke/.predict), `wrap_llamaindex_llm`
  (.complete), and `governed_tool` (screens an agent tool's model-facing text)
  make the common cases one line. No framework SDK imported at module load.
- `runtime/tools.py`: the doctor (content-blind, network-free environment /
  provider-key-presence / config diagnostics with a clear READY verdict) and
  the policy-pack registry (strict-logging, cost-guard, regulated-financial,
  self-hosted-gpu — named, testable governance bundles that are config, not
  code, with a real end-to-end effect).
- `scripts/release.py`: release hardening — CycloneDX SBOM, a dependency-
  surface audit proving the CORE is exactly one dependency (cryptography; the
  in-process moat requires a light core), and a deterministic content-blind
  file manifest (path/sha256/size) for signed attestation.
- Contract surface 89 -> 110; snapshot regenerated.

### The universality gate — GREEN
Every OpenAI-dialect provider in the catalog produces an adapter that passes
the FULL conformance kit. That is "any model" proven, not promised: not a list
we maintain, but a contract every provider satisfies, verified offline.

### Fixed — real design points surfaced by the battery
- Bedrock/Azure/OpenAI use deployment-supplied endpoints (gateway/deployment/
  SDK-default) the customer's client carries — catalog base_url is legitimately
  None for those; the well-formedness invariant reflects this.
- conformance kit closures fully typed; release-deps test import made hermetic
  (no scripts-package pollution).

### Verification
- New battery tests/test_runtime_universality.py: 28 checks — catalog
  well-formedness / unknown-provider / self-hosted-no-key / dialect-correct /
  async-openai-only; THE UNIVERSALITY GATE (all catalog adapters conformant) +
  report shape + broken-adapter-fails; frameworks (wrap_callable universal,
  langchain invoke/predict/reject, llamaindex complete, governed_tool
  screen/passthrough); doctor (ready/key-presence-content-blind/self-hosted/
  unknown/proof-without-signer); policy packs (populated/merge/strict-logging/
  applied-governs-e2e/unknown); catalog-adapter-full-pipeline; release
  manifest-deterministic-content-blind + core-minimal.
- 283 runtime tests green together; runtime mypy clean (25 files).

### === EXIT GATE: W0-W9 COMPLETE ===
- TRIPLE GATE x2: engine 1024/0 · plane 13/13 · runner 18/18 · GREEN both runs.
- The runtime is: universal (any model via one contract, any framework via
  the in-process wrap point, any environment incl. air-gapped), governed
  (secrets/PII/RBAC/rate/budget enforced in the call path), provable
  (cryptographic proof that verifies offline), resilient (breakers/bulkheads/
  chaos-tested/soak-proven), optimized (savings-parity-gated), extensible
  (signed plugins), standards-integrated (OIDC/OTel), adversarially audited,
  and release-hardened (SBOM + minimal core). Perfect per the plan's P1-P6.


## [Unreleased] — W8 COMPLETE: the platform layer (plugins, hot-reload, OIDC, telemetry)

### Added — the runtime becomes an extensible, standards-integrated platform
- `runtime/plugins.py`: signed-plugin loader on the SHIPPED Ed25519 signer.
  A plugin ships a canonical manifest (name/version/class/hook/order) + an
  Ed25519 signature; the registry recomputes the hash, verifies the signature,
  AND requires the key be on the trust list (a valid signature from an
  untrusted key is still refused). Plugins materialize as first-class kernel
  Engines: enforce-class plugins raise on the pipeline (fail closed, veto);
  observe-class plugins are isolated (a raise can never break a request). The
  signed `order` field is the true FIRING order per hook — after_response
  plugins are registered in reverse to fire ascending despite reverse unwind.
- `runtime/hotreload.py`: ReloadableConfig — atomic, versioned, validated
  config swap without restart. Readers always see a consistent snapshot
  (atomic reference swap, proven torn-read-free under 4 readers + writer);
  a rejected config is NOT applied (fail-safe, keeps last-good); subscribers
  notified after swap, outside the lock, with per-subscriber isolation.
- `runtime/oidc.py`: OIDCResolver — standards-based principal from a verified
  JWT. HS256 (HMAC) and EdDSA verification, expiry/nbf with leeway, issuer and
  audience enforcement, configurable principal/roles claims; unverified tokens
  REFUSED by default (opt-in only). bind() sets the shipped contextvar so the
  access engine and telemetry see the corporate principal. Content-blind:
  only principal + declared roles bound, never the raw token.
- `runtime/telemetry.py`: TelemetrySink contract + InMemorySink + OTelSink
  (OpenTelemetry spans/metrics: cost counter, token counter, latency
  histogram) + TelemetryEngine. Exports a projected CONTENT-BLIND record
  (id, model, provider, principal, tokens, cost, latency, verdicts, outcome —
  never payload). OTelSink degrades to a recorded no-op if the SDK is absent;
  sink exceptions are isolated — telemetry export can never break a request.
- Facade telemetry wiring (telemetry.enabled/otel + telemetry_sinks); contract
  surface 70 -> 89; snapshot regenerated.

### Fixed — real integration corrections caught by the battery
- TelemetryEngine and plugins must subclass the Engine base (handles_execution
  etc.) to register — corrected from plain classes.
- Plugin veto/isolation belongs in the ENGINE pipeline, not the observation
  hook bus (emit() swallows exceptions — an enforce plugin there would fail
  OPEN). Plugins are now Engines so enforce-raises propagate (fail closed).
- Signed `order` reconciled with reverse after_response unwind so it means
  actual firing order.

### Verification
- New battery tests/test_runtime_platform.py: 30 checks — plugin
  signed/trusted-loads, untrusted/tampered-manifest/tampered-sig/wrong-key
  REFUSED, observe-isolated, enforce-veto, signed-order firing, bad-declaration
  rejected; hot-reload atomic/versioned/notify/validation-fail-safe/
  concurrent-no-torn-read/subscriber-isolated; OIDC verified-resolve/wrong-
  secret/expired/issuer+audience/unverified-refused/missing-claim/bind-for-
  access; telemetry content-blind/per-request/error-path/sink-isolated/no-leak/
  otel-degrade; facade wiring; ALL-FOUR-COMPOSE + hot-reload-live-change.
- 255 runtime tests green together; runtime mypy clean (21 files).
- TRIPLE GATE x2: engine 996/0 · plane 13/13 · runner 18/18 · GREEN both.


## [Unreleased] — W0-W7 FULL ADVERSARIAL AUDIT: four runtime-level test suites (+49 tests)

### Added — the assembled runtime tested as an enterprise system under scrutiny
Four new suites treat the whole W0-W7 runtime (not one engine) as the unit
under test — the categories a real security review and SRE team run.

- `tests/test_runtime_security_redteam.py` (22): adversarial, organized by
  attacker goal — EXFILTRATE (secrets/PII/tool-args/error-text never reach any
  record, event, error, or proof; fingerprint is post-redaction),
  BYPASS (secret/access/budget block = zero provider calls; unicode-obfuscated
  key cannot slip a clean key through), FORGE (every proof field tampered ->
  rejected; swapped signature/public-key rejected; blocked calls still sealed;
  chain tamper localized), ESCALATE (rate limit EXACT under 200-thread flood;
  budget no-overspend under 64-thread flood; deny-by-default; auth not
  amplified), STARVE (1M-char payload, pathological unicode, dead provider,
  empty/whitespace — none crash or storm the pipeline).
- `tests/test_runtime_integration.py` (8): every engine active together —
  full-stack happy path asserting all invariants at once (engine order, real
  compression, provenanced cost, verdict ledger, verifiable proof, content-
  blind); cross-engine ordering (redact-before-compress, secret-block stops
  all downstream but proof still seals, cost on optimized tokens); failure
  propagation (provider-down sealed+attributed, transient recovered+priced);
  50-thread concurrent full-stack with no cross-contamination.
- `tests/test_runtime_traffic.py` (8): SRE-grade — sustained 5k-request
  throughput smoothness (p99 bounded vs p50), overhead baseline bounded,
  memory bounded at steady state past retention cap, no thread/fd leak,
  graceful drain completes in-flight + refuses new, provider outage->recovery
  smooth (breaker opens, fails fast, recovers clean), 300-thread facade
  traffic stable.
- `tests/test_runtime_regression_pins.py` (11): every real defect found across
  W0-W7 locked — W4 seam-order (cost sealed), W4 default-price-not-known-cost,
  W5 proof-verify-after-eviction, W6 bounded-chain, W6 auth-not-amplified,
  W7 candidates-is-list, W7 savings-parity, W3 streaming-no-seal, W4
  fingerprint-post-redaction, W2 unknown-error-retryable, W1 import-identity.

### Findings — audit surfaced THREE test-premise errors, ZERO code vulnerabilities
The red-team pass confirmed the runtime sound and corrected the tests:
- Secret firewall DOES catch the modern sk-proj- OpenAI key format; the
  entropy floor correctly rejects constant-fill placeholders (a feature, not
  a gap) — tests now use realistic high-entropy keys.
- SSN is defense-in-depth: BLOCKED by the secret firewall (HIGH) in block
  mode AND redacted by PII in off mode — never reaches a provider either way;
  tests aligned to the real two-layer behavior.
- TrustEngine seals outcome inside the chained body (ProofEngine surfaces it
  as a field) — tests assert the correct per-engine contract.

### Verification
- +49 runtime-level tests; 225 runtime tests green together; runtime mypy clean.
- TRIPLE GATE x2: engine 966/0 · plane 13/13 · runner 18/18 · GREEN both.


## [Unreleased] — W7 COMPLETE: optimization unification (OPT-1 contract + OPT-2 route planner)

### Added — one contract over the shipped savings, never reimplemented
- `runtime/optimize.py`:
  - OPT-1 Optimizer protocol + TieredOptimizer wrapping the SHIPPED
    compressors via the SHIPPED compose()/safe_compress(): tiers
    structural / salience / query / compose:a+b, each mapping to real code
    (StructuralCompressor, SalienceCompressor, QueryAwareCompressor). Query
    tier without a query degrades gracefully to structural; unknown tier
    raises; empty/tiny input safe. NoOpOptimizer is the honest floor
    (unchanged text, ratio 1.0).
  - OPT-2 RoutePlanner wrapping the SHIPPED Router: objectives cost
    (Router verbatim, max savings), quality (always capable), balanced
    (cheap only when Router-confident AND win-rate margin clears the band).
    Deterministic; writes an auditable RoutePlan.as_meta() (route_model,
    tier, reason, est_cost/saved, win_rate, objective).
- `runtime/optimization.py`: OptimizationEngine in the before-execution
  phase — AFTER security (only ever sees screened/redacted text), BEFORE
  economics (chosen model + shrunk payload inform cost). Compresses the
  payload the provider receives, sets meta.model from the route plan,
  refreshes the trust fingerprint to the SENT payload, and records
  content-blind optimization_events. Advisory to cost, never to
  correctness: a degraded compressor or unhelpful route lets the request
  proceed unoptimized.
- Facade wiring by config (optimization.enabled/compress/tier/route/
  objective/models); contract surface 64 -> 70; snapshot regenerated.

### The wave gate — SAVINGS-PARITY: GREEN
The tokens and cost the unified layer reports equal, to the unit, what the
shipped compressors and Router produce directly — structural, salience,
compose, and route savings all pinned identical. Optimization adds selection
and audit; it never changes the math. Shipped savings cannot regress.

### Fixed — latent bug caught in review
- RoutePlan.candidates had default_factory=dict for a List field — wrong
  type, would mis-initialize every plan. Corrected to list; mypy gate green.

### Verification
- New battery tests/test_runtime_optimization.py: 28 checks — protocol
  conformance, tier ladder (structural/salience/query-degrade/query/compose/
  unknown-raises/empty), 4 SAVINGS-PARITY gates, route objectives
  (cost/quality/balanced-margin), determinism, auditable meta, engine
  integration (compress-to-provider, model-from-route, content-blind events,
  fingerprint-follows-sent-payload, security-runs-first, tiny-skip, disabled-
  noop), facade wiring.
- Zero regression: 199 prior runtime-battery checks green; runtime mypy clean
  (17 files).
- TRIPLE GATE x2: engine 915/0 · plane 13/13 · runner 18/18 · GREEN both.


## [Unreleased] — W6 COMPLETE: survivability (REL-1..7) + 3 enterprise gaps closed

### Added — the request survives the provider
- `runtime/resilience.py`: composable, thread-safe primitives orchestrated by
  ResilientExecution (replaces the monkeypatch with an explicit wrapper):
  - REL-1 CircuitBreaker: per-provider closed/open/half-open, PER-REQUEST
    failure semantics, retry-storm ceiling (a downed provider is called at
    most `threshold` times, then refused — pinned).
  - REL-2 Bulkhead: per-provider concurrency isolation (a slow provider
    cannot starve others — pinned).
  - REL-3/3b BackoffPolicy: exponential + decorrelated jitter, capped;
    Retry-After honored over computed backoff (within cap).
  - REL-6 OutputValidator: declared-schema validation → typed OutputInvalid
    → bounded retry (repairs bad responses instead of returning garbage).
  - REL-4 ChaosInjector: deterministic delay/drop/error/malform faults;
    OFF = zero overhead (pinned); wired into the runner as `--chaos`
    (runner now 18/18).
  - Every degrade emits a stable reason on the SHIPPED degraded bus; every
    request records overhead to the SHIPPED overhead module.
- REL-7 soak/leak harness (scripts/soak.py): warms retention windows to
  steady-state, then measures RSS slope / fd / thread stability. Nightly 24h,
  CI short-window.

### Fixed — a REAL memory leak, caught by REL-7 soak, then proven fixed
- UNBOUNDED CHAIN GROWTH: TrustEngine and ProofEngine accumulated every
  sealed entry in memory forever — a genuine leak for a long-lived runtime.
  Now bounded rolling windows (default 5000); durable proof lives in the WORM
  sink/ledger. verify() anchors on the retained window's stored prev-hash
  after eviction. PROVEN: post-cap steady-state slope 0.00 KB/s over 10k+
  requests, entries bounded exactly at cap.

### Enterprise gaps closed (named in the W0–W5 honesty review)
- GAP #1 mypy: runtime package is mypy-clean ("no issues found in 15 source
  files"); gated by a battery test; strict-on-new-code policy now enforced.
- GAP #2 streaming: DOCUMENTED + TESTED architectural decision — a STREAMED
  request bypasses the after-phase, so it produces NO cost record and NO
  proof seal (we refuse to fabricate mid-stream totals); security STILL gates
  before the stream opens.
- GAP #3 overhead baseline: every resilient request records to overhead;
  p50/p99 now measurable (battery asserts p99 < 100ms and samples present).

### Verification
- New battery tests/test_runtime_resilience.py: 24 checks — breaker
  trip/half-open/reopen/storm-ceiling, bulkhead isolation+full+slow-provider,
  backoff jitter/Retry-After/applied-between-retries, validation reject/
  exhaust/exception-typed, chaos off-zero-overhead/error/delay/malform/
  rate-zero/survived-by-fallback, auth-never-retried, overhead baseline,
  1000-concurrent, streaming-skips-proof, mypy-clean gate, bounded-retention
  (trust + proof).
- Contract surface 54 → 64 (resilience exports); snapshot regenerated.
- TRIPLE GATE ×2: engine 887/0 · plane 13/13 · runner 18/18 (+chaos) · GREEN both.


## [Unreleased] — W5 COMPLETE: the proof spine (TRUST-1/2/3/4)

### Added — "we logged it" becomes "here is cryptographic proof"
- `runtime/proof.py`:
  - TRUST-1 ProofEngine: bridges kernel Trust residue into the SHIPPED
    hash-chained AuditLog (one chain of custody — extends the real ledger,
    never a shadow), registered FIRST so it unwinds LAST and seals the
    complete record (verdicts + cost + outcome); mirrors fail-open to the
    ledger and an optional WORM sink; seals the ERROR path too.
  - TRUST-2 Merkle tree (pure stdlib, domain-separated leaf/node hashing):
    merkle_root, merkle_proof, verify_merkle_proof + engine inclusion_proof
    / merkle_root — inclusion proofs verify OFFLINE.
  - TRUST-3 FileAuditSink (WORM): append-only NDJSON with per-line chained
    leaf hashes, fsync'd, reopen-recovers-head, verify() localizes any
    out-of-band edit.
  - TRUST-4 prove(request_id) → signed ProofPacket + verify_proof_packet,
    an OFFLINE verifier needing only stdlib + cryptography (no tokeymeter).
- Facade: `Runtime(config={'trust':{'proof':{'enabled':True}}}, signer=...)`
  swaps ProofEngine into the trust slot; `runtime.prove(request_id)` mints
  packets. Everything content-blind (fingerprints/verdicts/cost/model —
  never payload).
- Contract surface 46 → 54 (proof exports); snapshot regenerated (reviewed).

### The wave gate — GREEN
A signed proof packet + the offline verifier run on a CLEAN INTERPRETER
subprocess that never imports tokeymeter (only stdlib + cryptography):
VERIFIED for a genuine packet, CORRECTLY_REJECTED for a tampered one. The
proof outlives the runtime — the regulator's machine needs nothing from us.

### Verification
- New battery tests/test_runtime_proof.py: 21 checks — Merkle
  root/inclusion-all-positions/single-leaf-tamper/wrong-sibling; kernel
  residue seal+verify+localized-tamper; shipped-ledger mirror; content-blind
  residue; error-path seal; WORM append/reopen/out-of-band-edit-detect/
  engine-mirror; prove unsigned-rejected/signed-verifies/tampered-fails/
  missing-id/content-blind; engine inclusion roundtrip; facade
  requires-spine + end-to-end prove; CLEAN-INTERPRETER wave gate (+ tamper).
- Zero regression: 126 prior runtime-battery checks green.
- TRIPLE GATE ×2: engine 863/0 · plane 13/13 · runner 17/17 · GREEN both.


## [Unreleased] — W4 COMPLETE: the enforcement spine (GOV-1/2/4/5/6 + ECON-1/2/3)

### Added — enforce in the call path, seal in the proof
- `runtime/enforcement.py`:
  - SecurityEngine (GOV-5): secret firewall (block) → PII redaction → content
    terms (block), PRE-execution, strict. Redaction rewrites payload AND
    structured messages, then REFRESHES the trust fingerprint — no hash of
    secret/PII-bearing text ever enters trust residue (wave gate 2, pinned).
  - AccessEngine (GOV-1): RBAC/ABAC role→{models,actions}, DENY BY DEFAULT —
    unknown principal or unlisted model is a block, not a shrug.
  - RateLimitEngine (GOV-2): sliding-window requests/min + tokens/min per
    principal, thread-safe, Retry-After on the typed error.
  - CheckpointHook + LocalApprover (GOV-4): human-in-loop contract, engine
    side; approve/deny/pending with content-blind summaries; pending PARKS
    (never silently proceeds). Plane wiring parked D3.
  - Typed enforcement errors, all NOT retryable (auth/malformed semantics).
- `runtime/economics.py`:
  - ECON-1 kernel cost trace: reported-usage precedence, then response usage,
    then estimate. A KNOWN cost requires REAL provenance ('registered'/'list')
    — the pricing module's generic 'default' fallback is a GUESS, so
    cost_usd is None and the guess is kept as cost_estimated_usd with its
    source. NEVER zero, NEVER a guessed dollar on the receipt.
  - ECON-2 strict budgets via shipped keys (fingerprint-only, month-roll,
    concurrency discipline inherited): enforce | soft | approval.
  - ECON-3 approval bridge: budget exceed → GOV-4 checkpoint.
- GOV-6 verdict ledger: every engine appends content-blind policy:verdict
  pairs; TrustEngine seals them — INCLUDING ON THE ERROR PATH (a blocked
  request still leaves proof; on_error seals the ledger). Trust cost sealed too.

### Fixed — real defects caught by the battery (regression-pinned)
- SEAM-ORDER BUG: TrustEngine sealed before EconomicsEngine wrote cost
  (after_response unwinds in reverse). Trust now registers FIRST so it
  unwinds LAST and seals the COMPLETE record — verdicts + cost. Seam-order
  law now explicit in the facade.
- COST PROVENANCE: a generic-fallback price must not be trusted as a known
  cost by budgets/chargeback; engine now yields cost_usd=None for non-real
  provenance (receipt honesty enforced at the source, not patched at render).

### The wave gates — all GREEN
1. PDF p.13 VERBATIM: "confidential" content policy blocks the prompt AND a
   sealed, chain-verifying, content-blind audit entry records the block.
2. No secret/PII hash in trust residue (fingerprint after redaction).
3. Budget concurrency: 16 threads, breaker doctrine honored, every thread
   accounted, breach latches — no silent overspend.

### Verification
- New battery tests/test_runtime_enforcement.py: 34 checks — secrets
  block/content-blind, redaction-before-fingerprint, PII in payload+messages,
  PDF-p13 gate, 5-row RBAC deny-by-default matrix, rate-limit trip/reset/
  token/per-principal/16-thread-exact/error-seal, cost math/reported-
  precedence/unknown-never-zero/estimated-source, budget enforce/soft/
  approval(approve,deny,pending)/concurrency, full-spine ordered verdict
  ledger, facade config-wiring + real-dollar receipt + unpriced-dashes.
- Contract surface 33 → 46 (enforcement + economics exports); snapshot
  regenerated (reviewed).
- TRIPLE GATE ×2: engine 842/0 · plane 13/13 · runner 17/17 · GREEN both.


## [Unreleased] — W3 COMPLETE: Runtime facade + wow receipt (K4) + async kernel (KA-1)

### Added — the product face
- `runtime/facade.py`: `from tokeymeter import Runtime` —
  `Runtime(client=...)` (OpenAI/Anthropic duck-detected, async auto-routed),
  `Runtime(call=fn)` (any callable is a provider — the in-process
  guarantee), or explicit adapter. execute()/aexecute() run the full kernel
  pipeline (governance → reliability → trust → execution) over the shipped
  wrapper machinery; sync + async streaming; kernel LRU off by default
  (wrapper caches — no double-cache); `runtime.last` exposes the
  content-blind trace for inspect (W9).
- THE WOW RECEIPT: first call, TTY only — Routed / Cache / Cost / Saved /
  Compression / Latency / Verified / Trace. HONESTY LAW PINNED: every
  figure reads a real counter or event; absent source renders `—`
  (a callable provider shows `Cost —`, never an invented $); payload text
  never appears; TOKEYMETER_QUIET=1, non-TTY, or receipt="never" suppress.
- KA-1: `Kernel.aprocess()` + `ExecutionEngine.aexecute()` — native
  `ainfer` awaited, sync adapters offloaded to a worker thread (the loop is
  never blocked); identical drain semantics (in-flight counter spans the
  await; KernelStopped refuses new work). Sync pipeline refactored onto
  shared `_build_ctx`/`_finish_response` helpers — one source of truth.
- Demo rewired: `tokeymeter demo` now ends with the live kernel segment
  (Runtime → engine trace → trust-chain verify), still offline/no-keys.
- README "One line: the Runtime" quickstart — executable, and EXECUTED:
  the block runs verbatim in CI (the W3 wave gate).
- Contract surface 31 → 33 (Runtime, RuntimeConfigurationError); snapshot
  regenerated as a reviewed change.

### Verification
- New battery tests/test_runtime_facade.py: 15 checks (pipeline e2e,
  provider-required, client autodetection incl. reject, exceptions surface
  unchanged, sync+async streaming, receipt==counters + dash-for-absent +
  content-blind + quiet/non-TTY/once-only, async≡sync parity, async drain
  under load, mixed sync/async on one kernel, README-executes gate).
- TRIPLE GATE ×2: engine 808/0 · plane 13/13 · runner 17/17 · GREEN both.


## [Unreleased] — W2 COMPLETE: real adapters (K3) + error taxonomy (EXEC-4) + tool calls (EXEC-6)

### Added
- `runtime/adapters.py`: OpenAIAdapter / AnthropicAdapter / AsyncOpenAIAdapter
  wrapping the SHIPPED wrappers — kernel-path requests run the exact shipped
  pipeline (cache, compression, routing, audit, reported-usage truth) by
  construction. Async adapter routes through wrap() auto-detection so option
  defaults are always filled; sync infer() on the async adapter fails loud
  (async kernel path lands W3/KA-1). Structured messages via metadata;
  provider kwargs passthrough; streaming passthrough per wrapper design.
- `runtime/errors.py` (EXEC-4): typed taxonomy — RateLimited / AuthError /
  TransientError / MalformedRequest / ProviderDown — classified by
  exception-name MRO walk + HTTP-status fallback, ZERO SDK imports
  (stdlib-only core holds). Retry-After parsed onto RateLimited (REL-3b
  seam). Typed errors carry provider/status/original-class-name only —
  never payload (L4 pin). Unknown → TransientError(retryable), the safe
  default, pinned.
- EXEC-6: tool-call passthrough — responses round-trip untouched;
  `meta.tool_calls` carries COUNT only; args never in trace (L4 pin).
- ReliabilityEngine now routes on `retryable`: non-retryable typed errors
  (auth, malformed) get ZERO retries and ZERO fallbacks.
- Contract surface deliberately expanded 21 → 31 names (lazy exports keep
  `import tokeymeter.runtime` feather-light); snapshot regenerated as a
  REVIEWED change.

### The wave gate — USAGE-TRUTH PARITY: GREEN
Kernel-path economics envelope (event stream: type, hit, model, tokens
in/out, cost) is EXACTLY equal to the direct-wrapper envelope — cold call
AND two-call cache sequence (miss, store, hit). The kernel adds structure,
never distorts the money.

### Fixed / learned (regression-pinned)
- wrap_async requires full option kwargs; only wrap() fills defaults —
  adapters must enter through wrap() (battery-caught).
- MEASURED, NOT ASSUMED: the shipped wrapper is fail-open and makes its own
  second inner call on provider exception; the reliability-routing test now
  asserts Reliability adds ZERO attempts over that measured baseline.

### Verification
- New battery tests/test_runtime_adapters.py: 21 checks (round-trips,
  health, stream order sync+async, async≡sync result parity, PARITY GATE ×2,
  7-row taxonomy table, Retry-After, no-payload-in-errors, rel-routes
  auth-never/transient-retries, tool-call round-trip + count-only + L4).
- TRIPLE GATE ×2: engine 793/0 · plane 13/13 · runner 17/17 · GREEN both.


## [Unreleased] — W1 COMPLETE: K2 engine folder layout (Perfection Plan doc #8)

### Changed — the physical division (Migration Plan step 1, executed)
31 top-level modules + 4 subpackages divided into `tokeymeter/engines/
{execution, optimization, reliability, economics, governance, knowledge,
trust}/` per the frozen map (docs/ENGINE_MAP.md). Splits: integrations/
reconcile → economics; backends/cipher → trust; backends/redis_store →
optimization. Every old import path remains valid FOREVER via identity
alias shims (`sys.modules[old] IS canonical`) — imports, from-imports,
monkeypatching, and pickling are path-agnostic. Kernel-core untouched at
top level (decorator, _api, events, utils, paths, admin, demo, metrics,
runtime/). Inside engines/, all cross-references are canonical (no
shim-through-shim); staying modules keep their original imports and
resolve through shims with zero edits — the 1,600-line decorator was not
touched.

### Fixed — defects caught by the migration gates (regression-pinned)
- MULTI-DOT LAZY RELATIVE: keys.py function-body `from .content.secrets
  import` escaped the single-segment rewrite regex and resolved against
  economics. Canonicalized; rewrite rule upgraded.
- WHOLE-PACKAGE SIBLINGS MANGLED: audit/ and content/ internal `.log`/
  `.secrets` relatives were wrongly rewritten to top-level; restored
  canonical within-package.
- SPLIT-SIBLING POISON: `from tokeymeter.integrations import reconcile`
  forms cycled through the mid-executing shim; rule upgraded to "inside
  engines/, every old-path reference goes canonical, no exceptions".
- SIGNED MANIFEST INVALIDATED BY DESIGN: the release-integrity check
  correctly flagged the moved files; manifest regenerated + re-signed;
  release-check fixture path updated to the canonical audit home.

### Verification (W1 gate)
- New battery tests/test_engine_layout.py: 25 checks (13 alias-identity
  pairs across all engines + split children, no-duplicate-objects scan,
  both-path from-imports, bidirectional monkeypatch propagation, backends
  public names, split-package attribute access, pickle qualname survival,
  engines-init laziness, shim idempotence, top-level API intact,
  cold-import <2.0s bound (recorded; pre-move baseline not captured —
  noted honestly), stdlib-only core, ENGINE_MAP cross-check).
- Full module sweep: all 106 modules import on BOTH paths.
- TRIPLE GATE ×2: engine 772/0 · plane 13/13 · runner 17/17 · GREEN both runs.


## [Unreleased] — W0 Foundation Seal (Perfection Plan doc #8) + W1 Architecture Review

### Added — the Exit Gate machinery
- `scripts/run_perfection_gate.py`: machine-runs doc #8 P1–P6. Honest by
  construction: checks whose machinery doesn't exist yet report PENDING(wave),
  never green. v0 live checks: stdlib-only import guard, engine suite, plane
  suite, full runner, contract-surface freeze. First run: **GATE GREEN 5/5**
  (engine 747/0, plane 13/13, runner 17/17).
- `tests/test_contract_surface.py` + `tests/contract_snapshot.json`:
  public runtime contract frozen at 21 names; any surface change fails CI
  until deliberately regenerated — every contract change is a REVIEWED change
  (Exit-Gate P1, deprecation policy DEC-3).
- Coverage baseline measured and recorded (`scripts/perfection_baseline.json`):
  **67% line coverage** — the honest starting distance to the P4 ≥90% bar
  (enforced from W6).
- PLAT-1: CI engine job matrix widened to {ubuntu, windows, macos} ×
  CPython {3.11, 3.12, 3.13}; full-runner job stays OS-matrix on pinned 3.12
  (coverage where it matters, CI-minutes discipline where it doesn't).
- PLAT-2: `py.typed` shipped in the wheel (PEP 561); mypy scaffold in
  pyproject — lenient on legacy, `disallow_untyped_defs` on
  `tokeymeter.runtime.*` (strict-on-new-code policy).

### W1 (K2 engine layout) — Architecture Review artifact (L7 step 1 of 3)
Cross-import scan of the 33 to-be-moved top-level modules + 4 subpackages:
**37 cross-moved edges across 17 modules; 25 relative-import lines.** Hubs:
`pricing` (7 inbound), `degraded` (6 inbound); densest cluster is
optimization (as designed). Rewrite surface is bounded and mechanical;
alias-shim plan from doc #7 §2.2-K2 stands unmodified. Implementation is the
next action; battery `test_engine_layout.py` (≥14 checks) specced.


## [Unreleased] — K-track, Sprints 1+2: Runtime Kernel & provider adapters

### Added — tokeymeter/runtime: the kernel spine (pure stdlib, additive)
Compatibility mode per the Universal Runtime migration plan: nothing existing
is touched; engines DELEGATE to shipped modules. New package:
- `config.py` layered RuntimeConfig (defaults ← dict/JSON file ← TOKEYMETER_*
  env, JSON-coerced), dotted-path get.
- `container.py` DI container (instances, lazy singleton factories,
  test-seam override).
- `bus.py` HookBus with TWO emit semantics, and this split is load-bearing:
  `emit()` = observability, safe-failover (a raising plugin is reported on
  the degraded bus as `runtime_hook_error` and skipped — a bad plugin can
  never crash the core); `emit_strict()` = ENFORCEMENT, exceptions propagate
  (a governance veto fails CLOSED).
- `engine.py` Engine base + ordered registry (duplicate names rejected,
  exactly one handles_execution).
- `kernel.py` lifecycle (start → process* → shutdown with in-flight drain and
  bounded drain_timeout_s; new work refused with KernelStopped) and the
  pipeline: hooks → engines.before_request (order) → execute →
  after_response (REVERSE order) → hooks; on_error hooks then fail-loud.
  CONTENT-BLIND TELEMETRY: traces carry request_id/model/principal/engine
  timings and a sha256 payload fingerprint — never payload text (pinned).
- `providers.py` ProviderAdapter contract (get_info/infer/stream_infer/
  health_check), CallableAdapter wrapping the caller's own in-process client
  (no proxy, no hop — the moat holds), OpenAI/Anthropic contract stubs,
  ExecutionEngine with model→adapter routing.
- `engines.py` thin stages: Governance (stamps shipped identity principal;
  strict governance_check veto surface), Cache (CachePolicy contract +
  bounded thread-safe LRU default; hit short-circuits before the adapter),
  Trust (tamper-evident hash-chain of content-blind execution residue;
  verify() localizes the first bad index), Reliability (bounded retries +
  ordered model fallback around execution; kernel itself stays fail-loud),
  Knowledge (STAGE-GATED interface slot, no-op default — content-sovereign
  retrieval is deliberately NOT built at this stage).

### Fixed — two real defects caught by the new battery before they shipped
- ENFORCEMENT FAILED OPEN: governance_check originally ran through the
  safe-failover emit, silently swallowing policy vetoes. Enforcement and
  observability now have distinct emit semantics (regression-pinned).
- DEFAULTS POLLUTION: RuntimeConfig shallow-copied module _DEFAULTS and the
  env layer mutated shared nested dicts in place — one process setting
  TOKEYMETER_CACHE__ENABLED=false disabled caching for every later config.
  Defaults are now deep-copied (regression-pinned).
- TEST PORT COLLISION (plane): tokenet test_incidents bound PORT 8811,
  colliding with test_approval CAP_PORT 8811 — deterministic EADDRINUSE in
  full-suite runs. Moved to 8814.

### Verification
- New battery `tests/test_runtime_kernel.py` — 31 checks (config layering/
  isolation, DI, bus ordering + failover semantics, registry, pipeline order
  and unwinding, drain-under-load, content-blindness of trace AND trust
  chain, adapter contract + health checks, routing, cache short-circuit/
  LRU/disable, chain tamper localization, retry/fallback/exhaustion,
  knowledge no-op, full-stack integration).
- Zero regression: engine 746 passed / 1 skipped; tokenet 13/13; full
  system runner 17/17. `import tokeymeter.runtime` pulls in zero
  third-party modules — the stdlib-only core constraint holds.


## [Unreleased] — Phase-1 hardening, item 3: property-based fuzzing

### Added — hypothesis fuzzing of the five untrusted-input surfaces
hypothesis added to the dev extra (pure-Python, verified Windows/py3.14-safe;
profile 'tokeymeter' in conftest: max_examples=200, deadline=None so property
tests never flake under CI/Windows load). tests/test_fuzz_properties.py (14
properties) pins, for ALL generated input rather than hand-picked cases:
- SECRET SCANNER (security-critical): never raises on arbitrary text OR
  lossily-decoded bytes; result stays within the 16MB byte cap; every finding
  span lies inside the input; NO finding's recordable descriptor carries a raw
  secret value; planted real-shaped secrets are detected position-independently
  without leaking. Escalation run: 4000 generated + a pathological corpus
  (zero-width-split keys, null bytes, 100KB key runs, emoji/format-string
  storms) — all invariants held.
- ENVELOPE: wrap→unwrap is identity for any JSON-able payload (incl. with a
  future TTL); non-envelope values pass through unharmed.
- ID GRAMMAR (shared by identity/keys/ontology): anchored full-string match,
  length- and charset-bounded; accept and reject proven in both directions.
- safe_compress: never raises on any text (incl. against a compressor that
  always throws); ratio ∈ (0,1]; verdict fields always legal; fail-open
  preserves the original text.
- CACHE KEYS: deterministic (64-hex sha256); equal keys IFF equal prompts;
  suffix/prefix/whitespace/model/kwarg structural variants never collide.

No production defects surfaced — the invariants were already upheld; fuzzing
converts that from belief into a machine-checked contract before PyPI exposes
these surfaces to untrusted input.

## [Unreleased] — Phase-1 hardening, item 2: keys (rollover + concurrency)

### Fixed — one real latent hazard
- DEADLOCK HAZARD: the soft-band warning was emitted while holding the keys
  module lock; a degraded-bus subscriber calling back into any keys API
  (e.g. key_status) would deadlock on the non-reentrant lock. The warn is
  now DECIDED under the lock (exactly-once preserved via the band set) and
  EMITTED after release. Regression-guarded by a nosy-subscriber test.

### Contract corrected and pinned — enforcement is a circuit-breaker
- The originally enumerated "overshoot ≤ one call" bound was wrong for the
  actual (correct) fail-open design: the check never blocks the call path
  and cost is unknowable pre-call, so calls already in flight at breach
  complete and accrue. The TRUE contract, now in the check_current
  docstring and pinned by tests: sequential overshoot ≤ one call; K-way
  concurrent overshoot ≤ K in-flight calls; from the first check AFTER
  breach every call is refused (breach latch — verified under a 32-thread
  stampede). The cap is a spend circuit-breaker, not a reservation ledger.

### Pinned (tests/test_keys_hardening.py, 12 tests)
- UTC month rollover resets spend/calls/warned-bands and re-arms the cap:
  proven through the primitive across the Dec→Jan YEAR boundary, through
  every public path (check / accrual / status), and end-to-end through the
  real decorator (blocked "in July" → succeeds "on Aug 1").
- 32-thread accrual: call count exactly correct (lost-update detector),
  spend equal to the arithmetic sum (rel 1e-9).
- Soft-warn exactly once when 32 threads cross the band simultaneously.
- Re-register mid-month preserves accrual AND warned bands; a raised cap
  admits immediately, a lowered cap refuses immediately; rollover after
  re-register clears everything.

## [Unreleased] — Phase-1 hardening, item 1: async-path parity

### Fixed — two real engine defects surfaced by the new parity suite
- STREAMING KEY-BUDGET GAP: both `cache_stream` compute sites (normal L3 and
  high-stakes) lacked the hard key-budget stop — a capped key could stream
  past its budget. `_keys.check_current()` now fires BEFORE the first chunk
  at both sites; a stream against an exhausted cap raises at first iteration
  with zero chunks produced (pinned by test).
- UNRETRIEVED LEADER-FUTURE: when the async in-process single-flight leader
  failed with no followers waiting, its `set_exception` future was never
  retrieved — asyncio flags those at GC, polluting logs and hard-crashing
  deployments running warnings-as-errors. The exception is now marked
  retrieved at the set site; waiting followers still receive it unchanged
  (awaiting a done future re-raises).

### Contracts pinned (tests/test_async_parity.py, 12 tests)
- Async miss stamps principal into CallRecord AND CacheEvent; key_name
  stamped; hard cap raises before the awaited compute; accrual counts only
  real misses (hits/shadow never); asyncio.gather with two principals shows
  zero cross-bleed; streaming records carry principal+key_name; streamed
  exact hits replay without accrual; set_reported_usage lands
  token_source=reported through async AND stream paths.
- Leader-failure semantics DOCUMENTED AND PINNED as designed: followers
  fail open (bounded takeover retries, everyone surfaces the exception,
  nobody hangs), and no exception-holding future is ever left unretrieved.
- Enforcement scope contract (keys.py docstring): caps act at every METERED
  compute site — six sites total; `enabled=False` bypasses everything
  including budgets by definition; composition wrappers (with_memory)
  neither record nor enforce.
- Test-hygiene fix: the parity fixture now restores the library-default
  sync-to-file savings mode on teardown (exposed a latent cross-module
  ordering dependency in the compression suite).


## [Unreleased] — secret firewall hardening (found by Windows/3.14 stress run)

### Fixed — position-independent secret detection under time pressure
- The structured-secret firewall used a 0.5s wall-clock budget as a
  backtracking safety net, but on slower/Windows hardware an honest linear
  scan of a multi-MB payload could hit that budget BEFORE reaching a real key
  further in, silently missing it (stress check 2d: caught=False). Two fixes:
  (1) a cheap whole-input literal pre-scan now finds any present marker
  regardless of position and scans a bounded window around each hit — a
  present key is caught even under an abusively tight budget; (2) the
  wall-clock net raised to 5s (it was never the guarantee — the 16MB byte cap
  is the deterministic bound). Net effect: strictly stronger detection. No
  false positives added; full secret suite + red-team file still green.


## [0.14.0] — 2026-07-03  ·  Engine completion (universal, in-kind)

### Added — REFLEX verbs (T3.2): enforcement beyond block
- `TokeNetClient` gains `throttle` and `route` policy kinds enforced IN THE
  CALL PATH: `PolicyThrottled` (per-policy sliding-window rate limit) slows a
  running agent before the model is called; `route` rewrites the model arg to
  a cheaper tier. Both emit content-blind `throttled` / `routed` records
  carrying the policy id. `set_policies()` added for air-gapped/test loops.
- FIX: the route path previously inspected the @cache wrapper's co_varnames
  (always `(args, kwargs)`) and silently never rewrote — now unwraps to the
  underlying function and routes via kwargs. Real latent bug, caught by the
  new reflex battery.

### Added — Budget-Enforced Keys (T3.3): metering becomes control
- `tokeymeter/keys.py`: `register_key` stores a SHA-256 FINGERPRINT only
  (value used once in memory, never stored/logged/emitted; `key_name` is NOT
  emitter-whitelisted). `key()` context manager binds spend; a soft band
  emits one degraded warning, the hard cap raises `KeyBudgetExceeded` BEFORE
  the model call — enforced at all four decorator compute sites (sync/async ×
  miss/high-stakes). `leak_scan()` reuses the secret firewall's detectors and
  flags when a finding's fingerprint matches a REGISTERED key ("your
  prod-openai key is in .env.bak") — never returning the secret itself.

### Added — Context Passports (T3.4, emission half)
- `tokeymeter/context_passport.py`: content-blind provenance bus
  (fingerprint + source + before/after tokens + sensitivity label). A
  passport is auto-emitted whenever compression fires. Registry + flow
  policies are the plane half (N9.2).

### Added — Reported-usage truth (T1.1) + pricing-age discipline (T1.2)
- Provider `usage` fields flow through `set_reported_usage`; every record
  carries `token_source` = reported|estimated (reported beats estimate on
  misses). OpenAI + Anthropic wrappers already feed it.
- `pricing_age_days()` + `tokeymeter pricing` CLI surface how stale the LIST
  table is (registered/self-host rates are always current by definition).

### Verification
- 661 engine tests / 1 skipped; full system runner 18/18 with stress
  (adds reflex battery). New engine batteries: `test_engine_completion.py`
  (15), `test_reflex.py` (10). Wheel + sdist rehearsed in a clean venv.


## [Unreleased] — TokeNet control plane

### Added — N2 + T3.1 keystone: Org Registry + identity binding
- ENGINE `tokeymeter/identity.py` (v0.14): `set_principal` / `get_principal` /
  `principal()` context manager — contextvar-based, id-grammar-validated so a
  principal can never smuggle content; stamped into `CallRecord`,
  `CacheEvent`, and `DecisionRecord.from_event`; forwarded through the
  emitter whitelist (`principal` added to `_FIELDS`). 8-test suite pins
  scoping, restore-on-exception, thread + async isolation.
- PLANE `integrations/tokenet/registry.py` (pure): semantic edge rules
  (which kind may point at which), strict single-parent org tree with
  reach-the-company check, `manages` cycle detection, all-or-nothing spec
  validation, `resolve_chain` (agent→person→team→dept→company via runs_as),
  `has_authority` (approver manages subject or any ancestor; self-approval
  never authorized) returning the witnessing chain.
- Platform: `registry_seed` (upsert + structural-edge dedupe + seal),
  `registry_entity/resolve/orphans/authority`, cached structural index
  invalidated on seed, `_ensure_principals` observe+flag posture (unknown
  principal → one `unclaimed` agent node + one `unclaimed_principal` seal),
  `spend_rollup(by=principal|team|department)` — chargeback with
  'unattributed' never silently redistributed and totals conserved across
  groupings. `records.principal` column + index with a PRAGMA-guarded
  additive migration for pre-existing databases.
- Routes: POST `/api/registry/seed`; GET `/api/registry/entity|resolve|
  orphans|authority`, `/api/rollup?by=&days=`.
- Battery `tests/test_registry.py` — 37 checks including end-to-end wire
  proof: an engine-side `with tokeymeter.principal("dev")` call lands in the
  plane's records with its principal and rolls to the right team.

### Added — P2.2 Jira/Linear connector + N1 ontology (graph-native pattern-setter)
- `integrations/tokenet/ontology.py` — ONTOLOGY v1.0: 20 node kinds, 18 edge
  kinds (5 structural + 13 operational), loud validators, content-blind by
  construction (labels hard-capped, control chars rejected), round-trip-stable
  serialization. Every connector emits through it from day one.
- `connectors/base.py` — ActionFabricConnector: recursive content-blind gate
  (banned keys at any depth + string-length cap), idempotent retry/backoff
  delivery over injectable transports, constant-time HMAC-SHA256 inbound
  verification (fail-closed when unconfigured), fail-open outbound.
- `connectors/tickets.py` — Jira + Linear connectors (payload build, external
  id parse, inbound status normalization).
- `incidents.py` — pure incident model: deterministic idempotency key
  (tenant·kind·entity·period), validation, graph_records() emitting
  incident/ticket nodes + caused_by/opened_ticket edges.
- Platform: `graph_upsert_node/graph_add_edge/graph_counts` (ontology-validated
  store, tenant-isolated), incidents config, `open_incident` (dedupe → deliver
  → seal `incident_opened` → graph residue), `list_incidents`, `sync_incident`
  (+ seal `incident_synced`), `handle_ticket_webhook`; budget band ≥100%
  auto-opens a tracked incident (config-gated via `auto_open`).
- Routes: POST `/api/incidents/open` `/api/incidents/sync`
  `/api/incidents/config`, GET `/api/incidents` `/api/incidents/config`
  `/api/graph/counts`; public signature-verified
  POST `/api/webhooks/tickets?tenant=` (pre-auth, mirrors Slack).
- Battery: `tests/test_incidents.py` — 39 checks (ontology, content-blind
  gate, retry/fatal delivery, idempotent open, forged/unsigned/valid webhooks,
  sealing, tenant isolation incl. graph, auto-open trigger, end-to-end
  content-blindness); registered in `run_all_checks.py`.

All notable changes to Tokeymeter. Format: Keep a Changelog. Versioning: SemVer
(0.x — minor bumps may include additive API changes; breaking changes are
called out explicitly).

## [0.13.0] — 2026-07-02

### Added — pricing registry: the anti-fabrication layer
- `register_pricing(model, input_per_1m=, output_per_1m=)` — runtime price
  registration with validation; registered rates take precedence over the
  static list-price table and participate in longest-prefix matching.
- `register_selfhost_pricing(model, gpu_hour_rate_usd=, measured_tokens_per_second=)`
  — derives a self-hosted model's TRUE per-token rate from two measured
  inputs; the returned dict embeds the full derivation for auditability.
- `pricing_info(model)` / `estimate_cost_with_source(...)` — every rate
  resolves with provenance: `registered` / `registered_prefix` / `list` /
  `list_prefix` / `default`.
- Ledger provenance: `CallRecord.pricing_source` recorded per call
  (additive; pre-0.13 records classify against the current registry at
  report time).
- `savings_report()["pricing"]` honesty block: `all_priced`,
  `default_priced_calls`, `default_priced_usd`, `default_priced_models` —
  a USD figure resting on the generic fallback can no longer pass silently.
- Capacity view: `capacity_reclaimed(...)` (pure) and
  `tokeymeter.capacity_report(measured_tokens_per_second, gpu_hour_rate_usd=None)`
  — savings expressed in GPU-hours returned to the caller's own cluster;
  USD equivalence only when the caller supplies their own rate.
- `saved_input_tokens` / `saved_output_tokens` aggregates in the report.
- `tokeymeter demo` CLI subcommand — the self-hosted walkthrough
  (flagged fallback → derived true rate → honest savings + GPU-hours) in
  one command, offline, no keys.
- 19-test suite `tests/test_pricing_registry.py` pinning all of the above.

### Fixed
- `test_combinatorial` cross-pod single-flight simulation updated to pin an
  explicit shared namespace: the in-process pod factory collided with the
  (correct) namespace anti-collision guard, and the failure had been hidden
  behind optional-dep skip markers. Store-layer distributed locking was and
  is correct; the test now models real multi-process pods faithfully.
- `tokeymeter doctor` renders `n/a` instead of `None` for percentiles when
  no samples exist, and now leads with a one-line HEALTHY/DEGRADED verdict.

### Changed
- `dev` extra now includes `redislite` and `anyio` so the FULL suite —
  including cross-instance distributed-single-flight tests — always runs
  in CI. (This closes the skip-masking blind spot permanently.)

## [0.12.0] — 2026-06

Phase A reliability foundation: configurable TOKEYMETER_HOME + ledger
health with graceful degradation (P0.1); cross-platform unified check
runner and stress battery (P0.2); cache-hit overhead instrumentation,
buffered/in-memory ledger modes, `tokeymeter doctor` (P0.3); secret-firewall
performance guard — literal prefilter, chunked scan, wall-clock budget,
explicit scan cap, fail-closed/fail-open postures (P0.4).
