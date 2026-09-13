# Tokeymeter Versioning & Compatibility Policy

Enterprises adopt a dependency based on *upgrade risk*, not just features. This
document states exactly what we promise to keep stable, what we don't, and how
changes are announced — so you can pin, upgrade, and audit with confidence.

## 1. Versioning scheme (Semantic Versioning)

Tokeymeter follows [SemVer 2.0](https://semver.org): `MAJOR.MINOR.PATCH`.

- **PATCH** (`0.11.0 → 0.11.1`): bug fixes and internal changes only. No public
  API change, no behavior change that a correct caller would notice.
- **MINOR** (`0.11.0 → 0.12.0`): new, **additive** public API and features.
  Existing public API keeps working.
- **MAJOR** (`0.x → 1.0`, later `1.x → 2.0`): may contain breaking changes to the
  public API, announced and migrated per the deprecation policy below.

### Pre-1.0 status (honest disclosure)

Tokeymeter is currently **0.x (pre-1.0)**. Under SemVer, 0.x does not *guarantee*
a stable API across minor versions. In practice we already hold a stronger line:
the **public API surface is frozen by an automated test** (`FROZEN_EXPORTS` in
`tests/test_api_surface.py`), so a symbol cannot be added or removed without a
deliberate, reviewed change to that frozen set — the build fails otherwise. That
test is the enforcement mechanism behind every promise in this document.

**At 1.0**, this becomes a hard contract: no breaking change to the public API
within a major version, full deprecation windows, and persistent-format
forward-compatibility guarantees.

## 2. What "public API" means (covered by the guarantee)

The stability guarantee covers exactly the names exported from the top-level
`tokeymeter` package (`tokeymeter.__all__`, mirrored in `FROZEN_EXPORTS`) — the
decorators (`cache`, `cache_stream`), context managers (`lineage`,
`tenant_scope`), configuration calls (`set_default_store`, `enterprise_defaults`,
`set_event_preview_policy`, …), the store/compressor/memory classes, and their
**documented** keyword arguments.

For those, within a version line:
- documented keyword arguments keep their names, defaults, and meaning;
- documented return shapes remain stable (we may *add* keys to returned dicts —
  treat returned dicts as open for additive growth).

## 3. What is NOT covered (explicitly)

- **Anything prefixed with `_`** — `_get_default_store`, `_apply_compressor`,
  `_resolve_namespace`, etc. These are internal and may change in any release.
- **Exact log message text and `log.debug` output.** Use the degraded-event bus
  (`degraded_counts()`), not log scraping, for programmatic signals.
- **Internal store schema** beyond the documented persistent formats below.
- **Measured numbers** (savings %, latency) — these depend on your workload and
  are not API.
- **Behavior under explicitly-documented fail-open/fail-safe conditions** may be
  refined (e.g. adding a degraded event) without being considered a break.

## 4. Persistent / on-disk format compatibility

Persisted artifacts are explicitly versioned so a newer Tokeymeter can read older
data:

| Artifact | Version marker | Commitment |
|---|---|---|
| Cache envelope | `__tokeymeter_v1__` | A new envelope version is read alongside v1; old caches remain readable. |
| Cache backup (export/import) | header `version: 1` | `import_cache` reads its own and older format versions; legacy headerless backups still import (reported `unverified_legacy`). |
| Audit ledger (hash chain) | per-entry schema | Chain format is append-only and forward-readable; `verify_chain()` validates existing chains across upgrades. |

Rule: **a newer Tokeymeter reads data written by an older one.** Reading *newer*
data with an *older* library is not guaranteed (don't downgrade across a format
bump).

## 5. Deprecation policy

When a public API must change (post-1.0):
1. The old form keeps working and emits a `DeprecationWarning` naming the
   replacement.
2. It is documented as deprecated in `CHANGELOG.md` with the target removal
   version.
3. It is removed no sooner than **one MINOR release later** (1.x), or only at the
   next **MAJOR** for load-bearing API.
4. `FROZEN_EXPORTS` is updated in the same reviewed change that removes it, so the
   removal can never be silent.

Pre-1.0, we apply the spirit of this (announce in CHANGELOG, prefer additive
change) even though SemVer does not require it.

## 6. Upgrade-risk guidance

- **Pin** to a minor line in production (`tokeymeter>=0.11,<0.12`) and read the
  CHANGELOG before bumping minor.
- PATCH upgrades within a line are safe to take automatically.
- The `test_api_surface` frozen set is your machine-readable contract: if it
  hasn't changed between two releases, the public surface is identical.
- The signed integrity manifest (`integrity.verify_self()`) lets you confirm the
  installed package matches what was published — supply-chain assurance on top of
  API stability.

## 7. Where to look

- Public surface of record: `tokeymeter.__all__` / `tests/test_api_surface.py`.
- Change history: `CHANGELOG.md`.
- Security posture across versions: `docs/THREAT_MODEL.md`.
- Exception/observability contract: `docs/EXCEPTION_POLICY.md`.
