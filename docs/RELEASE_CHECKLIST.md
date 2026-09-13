# Tokeymeter Release Gate

A release to an institution must pass **every** item below. This gate exists so
that shipping is a deliberate, repeatable act — not "tests are green, push it." It
codifies the standards the rest of `docs/` defines. No item is skippable for a
release tagged for production/enterprise use.

Each item names the concrete artifact or command that satisfies it.

---

## 1. Correctness

- [ ] **Full test suite green** in a clean room (fresh extract → install → run):
      `pip install -e ".[dev]" && python -m pytest -q` → all pass, 0 failures.
- [ ] **Soak / concurrency gate green**: `python -m pytest -m soak -q`. For a
      tagged release run a **deep soak**: `TOKEYMETER_SOAK_SCALE=10 pytest -m soak`.
- [ ] **No skipped tests masking a regression** — every skip has a documented,
      environmental reason (e.g. optional `sqlite-vec` absent).
- [ ] **Clean-room reproduction**: the published artifact installs and passes the
      suite in an environment with nothing pre-cached.

## 2. Performance / no regressions

- [ ] **Benchmarks run** and compared to the previous release: cache-hit overhead,
      single-flight collapse, compression reduction (`benchmarks/`,
      `STRESS_TEST_REPORT`). No unexplained regression.
- [ ] **Bounded-resource invariants hold**: memory caps, LRU eviction, savings-log
      trimming, single-flight maps, session-lock LRU — confirmed by the soak gate.

## 3. Security

- [ ] **Threat model reviewed** (`docs/THREAT_MODEL.md`) — still accurate for this
      release; any new feature mapped to its threat category with residual risk
      stated.
- [ ] **Adversarial / fuzz input pass**: hostile inputs exercised — empty / None /
      unicode / oversized prompts, unserializable values, malformed backups,
      exotic argument types — none crash the caller, none produce a wrong answer.
      (Covered by `test_adversarial`, edge-case probes, restore-safety tests.)
- [ ] **Manual review of EVERY degraded / fallback / self-heal path**: each one
      (a) fails open at I/O boundaries or fail-safe in logic, (b) emits a degraded
      event (`degraded_counts()` enumerates all sources), and (c) never serves a
      wrong answer. Cross-check against `docs/EXCEPTION_POLICY.md`.
- [ ] **Secret hygiene**: no secret/key value in logs; key files created `0600`.
- [ ] **Signed integrity manifest**: `integrity.generate_manifest(signer=...)` +
      `integrity.write_manifest(...)`, and `verify_self(signer=...)` returns `ok`
      (not `unverified`). The release CI (`.github/workflows/release.yml`)
      generates the SIGNED manifest + SBOM and embeds them in the wheel. A release
      with only an unsigned manifest is **blocked**.
- [ ] **Tenant / isolation guarantees** intact — `tests/test_tenant_isolation.py`
      green; no new path bypasses namespace/lineage/tenant key composition.

## 4. API stability & versioning

- [ ] **Frozen public surface** reconciled: `tests/test_api_surface.py` passes; any
      intended addition/removal is a reviewed change to `FROZEN_EXPORTS`.
- [ ] **Version bump** follows `docs/COMPATIBILITY.md` (PATCH / MINOR / MAJOR) and
      is set consistently (`tokeymeter.__version__`, `pyproject.toml`).
- [ ] **Deprecations** (if any) emit `DeprecationWarning`, are listed in the
      CHANGELOG with a removal target, and respect the deprecation window.
- [ ] **Persistent-format changes** (envelope / backup / audit) bump their format
      marker and preserve "newer reads older".

## 5. Documentation

- [ ] **CHANGELOG.md** updated with the release's changes, grouped additive vs
      fixes vs breaking.
- [ ] **README / ARCHITECTURE** claims still true — especially any numbers
      (compression %, latency) re-stated as workload-dependent, not absolute.
- [ ] **Operator docs** current: configuration reference, failure-mode behavior,
      and the enterprise-hardening path (`enterprise_defaults`).

## 6. Supply chain

- [ ] **SBOM generated** and attached (release CI).
- [ ] **Dependencies reviewed** — no new runtime dependency added without
      justification; optional extras stay optional.
- [ ] **Reproducible build** — the wheel's manifest matches the source tree.

---

## Sign-off

A release is authorized only when every box above is checked. Record the
release version, the commit, the test/soak/benchmark results, and the manifest
signature in the release notes.

> Rule of thumb: if any degraded/fallback path can't be pointed to in
> `degraded_counts()`, or any public symbol changed without `FROZEN_EXPORTS`
> changing, the gate is **not** passed — investigate before shipping.
