# Tokeymeter Threat Model

This document states, honestly, what Tokeymeter defends against, by what
mechanism, what the **default** posture is, and — most importantly — what
**residual risk** remains. A threat model that only lists strengths is not useful
to a security reviewer; the residual-risk and out-of-scope sections are the point.

It covers the **Tokeymeter library** (in-process, local-first). The hosted
control plane (TokeyVue) has a separate, network-facing threat surface (authn/z,
multi-tenant backend, transport security) and is **out of scope here** because it
is not yet built; it will get its own model.

---

## 1. Scope and trust boundary

Tokeymeter runs **inside the application's own process** — there is no network
listener, no daemon, no separate service. It sits between the app and the LLM
provider as a function wrapper.

```
        [ application process  ── trust boundary ── ]
   app code ──> @meter wrapper ──> LLM provider API
                    │
                    ├─ cache (SQLite on-disk default at ~/.tokeymeter/cache.db;
                    │          in-memory fallback; Redis opt-in)
                    ├─ audit ledger (local, hash-chained)
                    └─ degraded/telemetry bus (in-process subscribers)
```

Consequences:
- Tokeymeter executes **with the application's privileges** and inside its trust
  boundary. It is not a sandbox and does not claim to defend the app from itself.
- The threats that matter are therefore about **what the optimization/proof layer
  itself could leak, corrupt, or get wrong** — not about an external attacker
  reaching a Tokeymeter network endpoint (there is none).

## 2. Assets

| Asset | Why it matters |
|---|---|
| Prompt & response **content** | May contain PII, secrets, proprietary data. |
| The **savings / proof record** (audit ledger) | The product's trust claim: every optimization is provable and tamper-evident. |
| **Secrets/keys** (HMAC install secret, signing key, cipher keys) | Compromise breaks confidentiality or proof integrity. |
| **Tenant/workload isolation** | One customer's cached answer must never reach another. |

---

## 3. Threats, mechanisms, and residual risk

Each item is tagged with the relevant STRIDE class.

### 3.1 Tenant / workload isolation — *Information Disclosure*
**Threat:** one tenant or workload receives another's cached answer for an
identical prompt+model.
**Mechanism:** the cache key is partitioned by **function namespace** (per
decorated function, with collision-safe disambiguation for dynamically generated
functions and entry-point salting for `__main__`), by **lineage** (task/
conversation boundary), and by **tenant** (`tokeymeter.tenant_scope(id)` or a
static `tenant=`), composed as the outermost key prefix. While a tenant or lineage
is active, **fuzzy semantic serving is disabled** (the semantic store is not
partitioned, so a fuzzy hit could bleed).
**Default posture:** per-function isolation is automatic. Tenant isolation
requires the app to declare the tenant — *which is unavoidable: the library cannot
isolate tenants it cannot distinguish.*
**Residual risk:** an app that runs multiple tenants through one function and does
**not** set `tenant_scope` will share cache across them. Mitigation: wrap requests
in `tenant_scope` (documented; surfaced in `enterprise_defaults()` guidance).
A future enhancement is a tenant-partitioned semantic store (today it is disabled
under tenant scope rather than partitioned).

### 3.2 Content confidentiality / disclosure — *Information Disclosure*
**Threat:** prompt/response content leaks through the cache file, telemetry, or
the audit record.
**Mechanism:** the audit ledger stores a **content-blind HMAC** of the prompt, not
the text. Event previews default to **hashed** correlation tokens, never raw
content. A **redactor** can strip PII before keying/serving, and under
`require_redaction` a redactor failure **fails closed** (the call errors rather
than leaking). Cache **values** can be **encrypted** (Fernet/AES) for SQLite/Redis.
**Default posture:** content-blind audit + hashed previews are **on by default**.
Value encryption and mandatory redaction are **opt-in**.
**Residual risk:** by default, cache **values** on disk (`~/.tokeymeter/cache.db`)
and in Redis are stored **in plaintext** — anyone with read access to that file or
Redis instance can read cached responses. RedisStore emits a **loud plaintext
warning** when no cipher is set. Mitigation: enable `require_encryption` (refuses a
no-op cipher) and a redactor via `enterprise_defaults(require_encryption=True)`.

### 3.3 Secret / key handling — *Information Disclosure, Spoofing*
**Threat:** HMAC/signing/cipher keys are exposed (logs, world-readable files) and
used to read content or forge proof.
**Mechanism:** the install secret and signing key are generated with
`secrets`-grade randomness and written **`0600`** (owner-only). Keys and secret
values are **never logged** (only file-handling errors and a deliberate plaintext
warning appear in logs).
**Default posture:** keys auto-created at `0600` under `~/.tokeymeter/`.
**Residual risk:** keys live on the local filesystem at rest; a host compromise or
an over-broad backup that copies `~/.tokeymeter/` exposes them. Tokeymeter does not
provide a KMS/HSM integration today. Mitigation: store the directory on encrypted
storage; supply externally-managed keys to the signer/cipher; exclude
`~/.tokeymeter/` from shared backups.

### 3.4 Audit tamper resistance & repudiation — *Tampering, Repudiation*
**Threat:** the proof record is altered, truncated, reordered, or a signer denies
having produced it.
**Mechanism:** the ledger is a **hash chain** — each entry carries a monotonic
`seq` and the `prev_hash` of its predecessor (genesis constant at `seq 0`), so any
edit, deletion, or reordering breaks the chain at that point and every point after.
`verify_chain()` detects it. **Checkpoints** record the chain head. Entries are
**signed**; under `require_audit_durability` no entry is silently dropped under
burst (bounded backpressure + synchronous write-through).
**Default posture:** HMAC signing (symmetric) gives **integrity** by default.
**Residual risk:** HMAC is **not non-repudiation** — the verifier holds the same
secret used to sign, so a holder of the secret could forge a chain. For regulated,
non-repudiable proof, enable `require_nonrepudiable_audit`, which **refuses to run
with HMAC** and requires an asymmetric `Ed25519Signer` (sign with a private key,
verify with the public key). Also: the ledger is local; off-host durability/escrow
(e.g. shipping checkpoints to WORM storage) is the operator's responsibility today
and a planned TokeyVue capability.

### 3.5 Injection — *Tampering, Elevation of Privilege*
**Threat:** attacker-controlled data alters a query or executes unintended code.
**Mechanism:** all SQL uses **parameterized queries**; the only interpolated SQL
fragments are **internal constants** (table names, generated `?` placeholders) —
never user input. Prompt content is treated as opaque bytes for hashing/keying,
not interpreted.
**Default posture:** no SQL-injection surface in the library.
**Residual risk:** **LLM prompt injection** — a malicious prompt manipulating the
*model's* output — is **out of scope**: that is a property of the application's
prompts and the provider, not of a caching/proof layer. Tokeymeter neither
introduces nor mitigates it. A custom `key_fn`/`encoder`/`redactor` supplied by the
app runs with app privileges and is the app's responsibility (Tokeymeter fails open
or closed around it per policy, but does not sandbox it).

### 3.6 Replay & ordering — *Tampering, Repudiation*
**Threat:** audit entries are replayed or reordered to misrepresent history.
**Mechanism:** monotonic `seq` + `prev_hash` chaining make order **integral to the
hash**; a replayed or reordered entry fails `verify_chain()`. Cache "replay" is not
a meaningful threat (a cache hit is deterministic by key).
**Residual risk:** within a single process, ordering is authoritative; **across
processes** sharing one ledger, concurrent appenders are serialized by the write
lock, but a clock-skew-based interpretation of timestamps (not the chain) could
mislead a human reader. The cryptographic order (seq/hash) remains sound.

### 3.7 Availability / denial of service — *Denial of Service*
**Threat:** the optimization layer degrades or hangs the host app under load.
**Mechanism:** **fail-open** everywhere (a broken store/cipher/redactor never
breaks the call); **bounded** memory (LRU caches, capped single-flight maps,
LRU-bounded session locks, capped fidelity log); **bounded** disk (savings log
trimmed to a rolling window); audit durability uses **bounded** backpressure, not
unbounded blocking.
**Residual risk:** under `require_audit_durability` with a genuinely stalled
flusher, the host thread performs a synchronous write (bounded latency, not a hang).
A pathologically slow user-supplied degraded **subscriber** runs in the calling
thread and can slow it — documented; keep subscribers cheap.

---

## 4. Hardening to enterprise posture

`tokeymeter.enterprise_defaults(...)` flips the safe posture in one call and is
**immediately satisfiable** (it never leaves calls failing closed for lack of
setup):

- hashed event previews + runtime guards (always),
- a redactor ensured + `require_redaction` (PII stripped or fail-closed),
- opt-in, **enforced** when enabled: `require_encryption` (no plaintext values),
  `require_keyed_cache` (HMAC'd distributed keys), `require_nonrepudiable_audit`
  (Ed25519, not HMAC), `require_audit_durability` (no dropped proof entries).

"Enforced when enabled" means construction **refuses to run unsafely** — e.g. a
RedisStore with no cipher will not start under `require_encryption`.

---

## 5. Explicitly out of scope

- **The host is trusted.** A compromised process/host that already runs the app
  can read app memory and the local cache/keys. Tokeymeter is not a sandbox.
- **LLM prompt injection** and model output safety — the app's and provider's
  responsibility.
- **Network transport** to the LLM provider — handled by the provider SDK/TLS.
- **TokeyVue (hosted control plane)** — separate network-facing model, not built.
- **Side-channel / timing attacks** on the cache — not defended against.

---

## 6. Residual-risk summary (the one-screen version)

| If you do nothing | You get | To close it |
|---|---|---|
| Multi-tenant through one function, no `tenant_scope` | cross-tenant cache sharing | wrap requests in `tenant_scope` |
| Default disk/Redis cache | **plaintext** cached values | `require_encryption` + cipher |
| Default audit signing | integrity, **not** non-repudiation | `require_nonrepudiable_audit` (Ed25519) |
| Default redaction | none (content may be cached) | redactor + `require_redaction` |
| Default durability | proof entries *may* drop under extreme burst | `require_audit_durability` |
| Keys on local disk | exposed by host compromise / broad backup | encrypted storage; external keys; exclude from backups |

The honest summary: **out of the box Tokeymeter is content-blind in its proof and
telemetry and isolates per function; the confidentiality of cached values, true
non-repudiation, mandatory redaction, and guaranteed durability are one
`enterprise_defaults()` call away — and are enforced, not merely advertised.**
