# Secret Firewall — Adversarial Stress & Red-Team Report

*Increment 1 of the NOVUE content governance layer. Production-readiness campaign.*

## Summary

The structured-secret firewall was subjected to a full adversarial battery across
seven disciplines: red-team evasion, security (ReDoS/exhaustion/malformed),
performance/load/spike/soak, globalization/localization, boundary/exploratory,
concurrency, and chaos fuzzing.

**Final result: 52 checks passing (14 unit + 38 adversarial), 0 failing.**
The campaign found **3 real defects**, all fixed; the locked engine suite remained
at 590/10 throughout.

## Defects found and fixed

1. **Glued-credential evasion (CRITICAL).** A secret with no surrounding word
   boundary ("xEjAKIAIOSFODNN7EXAMPLEOX") evaded the `\b`-anchored patterns. The
   chaos fuzzer surfaced **1,263 leaks** out of 5,000 random inputs. *Fix:* a
   second normalized scanning pass with boundary-tolerant variants of the
   high-value credential detectors (specific prefixes/lengths/checksums, so no
   new false positives), with offsets mapped back to the original text. Leaks
   went to **0**.
2. **Zero-width-character evasion.** A key split by a zero-width space evaded
   detection. *Fix:* the normalized pass strips zero-width / soft-hyphen
   characters before re-scanning. Now caught.
3. **Scanner-raising propagation.** A custom scanner that *raised* (rather than
   returning `scanned=False`) propagated the exception instead of failing closed.
   *Fix:* `enforce()` wraps the scan call; any raise in fail-closed posture blocks
   the call. Defense in depth.

Two test-fixture bugs were also corrected (unrealistic zero-entropy fake keys that
the entropy gate correctly rejected — the firewall was right, the test was wrong).

## What passed

- **Red team:** plain / embedded-in-large-doc / code-fenced / JSON-valued / glued /
  zero-width credentials all blocked; secret never leaks into the block exception;
  real key found amid 500 random near-miss tokens.
- **Security:** no ReDoS (worst pathological input ~50ms); 5MB input handled without
  crash; scanner never raises on malformed/hostile/unicode/surrogate input;
  placeholder-injection does not corrupt restoration.
- **Performance:** typical-prompt scan **p99 = 0.24ms** (budget < 5ms); ~6,000
  scans/sec; no perf drift over 50,000 scans (soak).
- **Globalization:** secrets caught amid CJK/Korean/Arabic/RTL/emoji-ZWJ text;
  offsets correct despite multibyte characters; international IBANs validated.
- **Concurrency:** no secret leaked and no cross-thread placeholder contamination
  under 200-way concurrent enforcement; scanner stable under 500 concurrent scans.
- **Chaos:** 0 crashes and 0 leaks over 5,000 random fuzzed inputs.

## Documented boundaries (Tier-1 limits — none are silent leaks)

- **Whitespace-fragmented credential** ("AKIA IOSF ODNN ...") is not reassembled.
  Mitigation: a high-secret-density block policy + later normalization tiers.
- **Homoglyph / full-width-digit** substitution of checksum identifiers is not
  normalized. Deferred to a later normalization tier.

These are acceptable for Tier 1: every realistic high-value leak path is caught,
and the boundaries are explicit (recorded in code and here), not hidden.

## Verdict

The firewall is production-grade for its scope: high precision (zero false
positives on the over-defense trap set), high recall on realistic credential leak
paths, evasion-resistant, fast enough to run on every call, thread-safe, and
fail-closed. It is ready to be wired into the engine call path, and ready for an
independent security review before a regulated-customer deployment.
