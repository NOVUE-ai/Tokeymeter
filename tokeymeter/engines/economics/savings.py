"""
Local savings tracker.

Every cache decision writes one JSONL line to ~/.tokeymeter/savings.jsonl.
Reports compute on demand from the file so they survive process
restarts and reflect actual persisted history.

Tracked dimensions:
  - hit_type: "exact" | "semantic" | "single_flight" | "shadow_*" | None
  - shadow:   True if this was a measurement-only lookup
  - tag:      workload label (e.g. "support", "rag")
  - model:    used for cost estimation

The report includes a dedicated `shadow` block with `would_have_saved_usd`
— what your live cache will save once you flip shadow=False.
"""
from __future__ import annotations

import atexit
import json
import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from tokeymeter import paths
from tokeymeter.engines.reliability.degraded import emit_degraded


@dataclass
class CallRecord:
    timestamp: float
    model: str
    hit: bool
    hit_type: Optional[str]
    input_tokens: int
    output_tokens: int
    estimated_cost: float
    latency_ms: float
    shadow: bool = False
    tag: Optional[str] = None
    # v0.6: compression metrics
    compression_ratio: Optional[float] = None
    tokens_saved_via_compression: int = 0
    compression_method: Optional[str] = None
    # v0.13: provenance of the rate behind estimated_cost — "registered",
    # "registered_prefix", "list", "list_prefix", or "default". Records at
    # "default" rest on the generic fallback, NOT a real price for that model;
    # report() surfaces them so a fallback-derived figure can never pass
    # silently as customer-specific truth. None on pre-v0.13 records.
    pricing_source: Optional[str] = None
    # v0.14: identity binding — the registered person/agent id (contextvar)
    # behind this call. None = unattributed; control plane surfaces those.
    principal: Optional[str] = None
    # v0.14 T3.3: local key identity bound during this call (name only —
    # values never enter the record stream; NOT emitter-whitelisted).
    key_name: Optional[str] = None
    # v0.14 T1.1: "reported" = provider's own usage counts; "estimated" =
    # chars/4 heuristic. Hits are estimated by construction (avoided cost).
    token_source: Optional[str] = None
    # v0.14 self-host: operator-declared endpoint id (contextvar or @cache arg)
    # this call executed against — a stable NAME ("vllm-a100-pool"), never a
    # URL/host/credential. None = unattributed; per-endpoint unit cost,
    # consolidation, and queue economics all key on this. Content-blind.
    endpoint_identity: Optional[str] = None
    # v0.14 self-host: milliseconds this MISS spent queued before execution, as
    # reported by the serving layer (vLLM/TGI). None = unknown (never a
    # fabricated 0). Hits don't queue, so this is None on hits by construction.
    # The measured input behind Queue Economics and SLO cost.
    queue_wait_ms: Optional[float] = None
    # v0.16 task boundary: WHICH unit of work this call belongs to — an
    # operator-supplied id bound by `tokeymeter.task(...)`. An agent task is
    # 10-100 calls; this is the field that makes the TASK the unit of
    # accounting instead of the request, which is what every other tool
    # measures. None = the call was not made inside a task boundary; those
    # group as unattributed exactly as principal=None does. Content-blind: an
    # identifier (ticket/job/run id), never content.
    task_id: Optional[str] = None
    # v0.16 task boundary: non-reversible "sha256:<12hex>" digest of the cache
    # key, which is itself a digest of the call arguments. A hash of a hash —
    # no content can survive it. Identical calls inside one task collapse to
    # the same value, which is how a loop is detected: the same fingerprint
    # repeating means the agent is asking the same thing and making no
    # progress. We compare hashes; we never read a prompt.
    prompt_fingerprint: Optional[str] = None
    # v0.16 task boundary: the KIND of task this call served ("support",
    # "extract"), bound by `tokeymeter.task(agent=...)`. task_id identifies the
    # instance; `agent` is what execution rules key on, so it must be on the
    # record or a proposed agent-conditioned rule cannot be SIMULATED against
    # history — the platform owner would be pushing exactly the rules they most
    # need to preview blind. Content-blind: an operator-supplied label.
    agent: Optional[str] = None
    # v0.16 progress signal: non-reversible digest of the RESPONSE. The prompt
    # cannot tell you an agent is stuck — a real agent carries its conversation
    # history, so every prompt hash is unique even when it is failing the same
    # way for the thirtieth time. The response can: a stuck agent gets the SAME
    # answer back. Progress is novelty in the OUTPUT. Content-blind: hashed,
    # never read, never stored.
    response_fingerprint: Optional[str] = None
    # v0.18 compliance context — DECLARED by the application, never inferred.
    # On the record because "the rule was enforced" is worth nothing to an
    # auditor without "on what, and where from". Both are operator labels
    # ('PHI', 'EU'), never content.
    data_class: Optional[str] = None
    region: Optional[str] = None
    # Which compliance rule(s) applied to this call, so a specific decision can
    # be traced back to the specific policy line that caused it.
    policy_rules: Optional[str] = None


def build_call_record(
    *,
    model: str,
    hit: bool,
    hit_type: Optional[str],
    input_tokens: int,
    output_tokens: int,
    estimated_cost: float,
    latency_ms: float,
    shadow: bool = False,
    tag: Optional[str] = None,
    compression_ratio: Optional[float] = None,
    tokens_saved_via_compression: int = 0,
    compression_method: Optional[str] = None,
    pricing_source: Optional[str] = None,
    principal: Optional[str] = None,
    token_source: Optional[str] = None,
    key_name: Optional[str] = None,
    endpoint_identity: Optional[str] = None,
    queue_wait_ms: Optional[float] = None,
    task_id: Optional[str] = None,
    prompt_fingerprint: Optional[str] = None,
    agent: Optional[str] = None,
    response_fingerprint: Optional[str] = None,
    data_class: Optional[str] = None,
    region: Optional[str] = None,
    policy_rules: Optional[str] = None,
) -> "CallRecord":
    """Single construction point for every CallRecord, whatever the code path.

    The decorator and the direct SDK-wrapper record paths BOTH build through
    here. This exists so a new ledger field is added in exactly one place: add
    it as a keyword here (and to the dataclass) and every path picks it up.
    Before this factory, the async OpenAI wrapper built CallRecord inline and
    silently omitted pricing_source / principal / key_name — the class of drift
    a shared factory makes structurally impossible.

    Callers pass ALREADY-RESOLVED values (this does no contextvar reads and no
    estimation) so the factory stays a pure, side-effect-free constructor.
    """
    return CallRecord(
        timestamp=time.time(),
        model=model,
        hit=hit,
        hit_type=hit_type,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_cost=estimated_cost,
        latency_ms=latency_ms,
        shadow=shadow,
        tag=tag,
        compression_ratio=compression_ratio,
        tokens_saved_via_compression=tokens_saved_via_compression,
        compression_method=compression_method,
        pricing_source=pricing_source,
        principal=principal,
        token_source=token_source,
        key_name=key_name,
        endpoint_identity=endpoint_identity,
        queue_wait_ms=queue_wait_ms,
        task_id=task_id,
        prompt_fingerprint=prompt_fingerprint,
        agent=agent,
        response_fingerprint=response_fingerprint,
        data_class=data_class,
        region=region,
        policy_rules=policy_rules,
    )


class SavingsTracker:
    """Append-only JSONL log of every cache lookup."""

    def __init__(self, path: Optional[str] = None,
                 max_bytes: int = 100 * 1024 * 1024):
        # path resolution: explicit arg > central resolver (env / set_home / default).
        # Resolving here means set_savings_path() takes effect via _rebind_tracker().
        self._path = os.path.expanduser(path) if path else paths.savings_path()
        self._lock = threading.Lock()
        # Bound on-disk growth: the savings log is a rolling record, so when it
        # exceeds max_bytes we trim to the most recent half (amortized O(1) per
        # write). Without this, a high-volume service would grow it without bound
        # and eventually fill the disk. report() reads the single file unchanged.
        self._max_bytes = int(max_bytes)
        self._bytes_since_check = 0
        # Crash-tail repair (see _ensure_clean_tail_locked): checked once per
        # tracker instance before the first disk append, so a torn last line
        # left by a killed writer can never swallow the next record.
        self._tail_verified = False
        # Re-check size after roughly 10% of the cap has been written, so the
        # overshoot beyond max_bytes is bounded to ~10% regardless of record size.
        self._check_interval_bytes = max(1, self._max_bytes // 10)
        # ── ledger health (P0.1) ──────────────────────────────────────────
        # The ledger must NEVER silently fail: a read-only home, missing $HOME,
        # or locked-down sandbox marks the ledger degraded and emits an event,
        # rather than crashing the caller or returning zero with no explanation.
        self._writable = True
        self._degraded = False
        self._records_written = 0
        self._write_failures = 0
        self._read_failures = 0
        self._last_error: Optional[str] = None
        self._last_error_type: Optional[str] = None
        # ── write modes (P0.3) ────────────────────────────────────────────
        # Default is synchronous (a write per call) — simple and durable. For
        # high-throughput or slow-disk environments, buffered mode batches writes
        # off the hot path; in-memory mode keeps records in RAM only (benchmarks,
        # read-only sandboxes). Both opt-in; both kept durable via flush + atexit.
        self._buffered = False
        self._in_memory = False
        self._buffer: list = []           # pending serialized lines (buffered mode)
        self._buffer_max = 128            # flush when this many lines accumulate
        self._mem: list = []              # record dicts (in-memory mode)
        self._mem_cap = 200_000           # bound RAM use in in-memory mode
        self._flush_interval = 0.0        # >0 -> background flush cadence (seconds)
        self._flusher = None
        self._closed = False
        self._ensure_parent()

    def _ensure_parent(self) -> None:
        """Best-effort creation of the ledger's parent dir. On failure, mark the
        ledger degraded and emit an event instead of raising into import/caller."""
        try:
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            with self._lock:
                self._mark_degraded_locked(e, writable=False, source="savings_dir")

    def _mark_degraded_locked(self, error, *, writable=None, source="savings_ledger"):
        """Record a ledger failure and emit a bounded degraded event. Caller holds
        self._lock. Emits on the first transition then every 256th failure, so a
        sustained outage can't flood subscribers."""
        was_healthy = not self._degraded
        self._degraded = True
        if writable is not None:
            self._writable = writable
        msg = str(error)
        self._last_error = (msg[:197] + "...") if len(msg) > 200 else msg
        self._last_error_type = type(error).__name__
        total = self._write_failures + self._read_failures
        if was_healthy or total % 256 == 0:
            emit_degraded(source, error, function_name="SavingsTracker")

    def ledger_health(self) -> dict:
        """Operational truth about the ledger: is it writable, has it degraded,
        how many records landed, how many failures, and the last error."""
        with self._lock:
            return {
                "path": self._path,
                "writable": self._writable,
                "degraded": self._degraded,
                "records_written": self._records_written,
                "write_failures": self._write_failures,
                "read_failures": self._read_failures,
                "last_error": self._last_error,
                "last_error_type": self._last_error_type,
                "buffered": self._buffered,
                "in_memory": self._in_memory,
                "buffer_pending": len(self._buffer),
            }

    def _ensure_clean_tail_locked(self) -> None:
        """One-time (per instance) crash-tail repair. Caller holds self._lock.

        A writer killed mid-write (SIGKILL, OOM, power loss) leaves a torn
        final line with no newline. A naive append would concatenate the NEXT
        record onto that fragment, corrupting BOTH — one crash silently costs
        a record. Before this instance's first disk append: if the file ends
        without a newline, write one, isolating the fragment on its own line
        (the tolerant reader already skips it). Our own writes always end with
        a newline, so one check per instance suffices. Never raises.
        """
        if self._tail_verified:
            return
        self._tail_verified = True
        try:
            if os.path.exists(self._path) and os.path.getsize(self._path) > 0:
                with open(self._path, "rb") as f:
                    f.seek(-1, os.SEEK_END)
                    if f.read(1) != b"\n":
                        with open(self._path, "a", encoding="utf-8") as f2:
                            f2.write("\n")
        except OSError:
            # Missing/locked file etc. — the append path handles its own
            # failures (degraded + memory fallback); nothing to do here.
            pass

    def record(self, rec: CallRecord) -> None:
        try:
            d = asdict(rec)
            if self._in_memory:                       # RAM only, no disk
                with self._lock:
                    self._mem.append(d)
                    if len(self._mem) > self._mem_cap:
                        del self._mem[: len(self._mem) - self._mem_cap]
                    self._records_written += 1
                return
            line = json.dumps(d) + "\n"
            if self._buffered:                        # batch off the hot path
                with self._lock:
                    self._buffer.append(line)
                    self._records_written += 1
                    if len(self._buffer) >= self._buffer_max:
                        self._flush_locked()
                return
            with self._lock:                          # synchronous (default)
                self._ensure_clean_tail_locked()
                with open(self._path, "a", encoding="utf-8") as f:
                    f.write(line)
                self._records_written += 1
                self._bytes_since_check += len(line.encode("utf-8"))
                if self._bytes_since_check >= self._check_interval_bytes:
                    self._bytes_since_check = 0
                    self._maybe_trim_locked()
        except OSError as e:
            with self._lock:
                self._write_failures += 1
                # Disk unwritable (read-only home, locked-down sandbox). Durable
                # writes are impossible here, so fall back to in-memory: keep the
                # metrics for this session and keep report() working, rather than
                # silently dropping them. Surface it as degraded so it's visible.
                if not self._in_memory:
                    self._in_memory = True
                    self._mark_degraded_locked(e, writable=False,
                                               source="savings_fallback_memory")
                else:
                    self._mark_degraded_locked(e, writable=False, source="savings_write")
                try:
                    self._mem.append(d)
                    if len(self._mem) > self._mem_cap:
                        del self._mem[: len(self._mem) - self._mem_cap]
                    self._records_written += 1
                except Exception:
                    pass
            # never crash the caller
        except (TypeError, ValueError) as e:
            with self._lock:
                self._write_failures += 1
                self._mark_degraded_locked(e, source="savings_serialize")
            # serialization bug, not a disk problem; still never crash the caller

    # ── buffered-mode plumbing (P0.3) ─────────────────────────────────────
    def flush(self) -> None:
        """Persist any buffered records to disk in one batched write. Safe to call
        anytime; a no-op in synchronous or in-memory mode."""
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        """Caller holds self._lock. Batches all pending lines into one append.
        On failure, keeps the buffer for retry (bounded) and marks degraded —
        never raises into the caller."""
        if not self._buffer:
            return
        try:
            self._ensure_clean_tail_locked()
            with open(self._path, "a", encoding="utf-8") as f:
                f.writelines(self._buffer)
            self._bytes_since_check += sum(len(p.encode("utf-8")) for p in self._buffer)
            self._buffer = []                          # clear only on success
            if self._bytes_since_check >= self._check_interval_bytes:
                self._bytes_since_check = 0
                self._maybe_trim_locked()
        except OSError as e:
            self._write_failures += 1
            self._mark_degraded_locked(e, writable=False, source="savings_flush")
            if len(self._buffer) > self._mem_cap:      # bound growth if disk stays dead
                self._buffer = self._buffer[-self._mem_cap:]

    def _start_flusher(self) -> None:
        if self._flusher is not None or self._flush_interval <= 0:
            return

        def _loop():
            while not self._closed:
                time.sleep(self._flush_interval)
                self.flush()

        self._flusher = threading.Thread(
            target=_loop, name="tokeymeter-savings-flush", daemon=True)
        self._flusher.start()

    def set_buffered(self, enabled: bool, buffer_size: int = 128,
                     flush_interval: float = 0.0) -> None:
        with self._lock:
            if self._buffered and not enabled:
                self._flush_locked()                   # drain on disable
            self._buffered = bool(enabled)
            if enabled:
                self._in_memory = False
            if buffer_size:
                self._buffer_max = int(buffer_size)
            self._flush_interval = float(flush_interval)
        if enabled and flush_interval > 0:
            self._start_flusher()

    def set_in_memory(self, enabled: bool) -> None:
        with self._lock:
            self._in_memory = bool(enabled)
            if enabled:
                self._buffered = False

    def _iter_records(self):
        """Yield record dicts from the durable store (flushing buffered writes
        first) or, in in-memory mode, from RAM. Single source of truth for report()."""
        if self._in_memory:
            for rec in list(self._mem):
                yield rec
            return
        self.flush()
        try:
            with open(self._path, encoding="utf-8") as f:
                for line in f:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            return
        except OSError as e:
            with self._lock:
                self._read_failures += 1
                self._mark_degraded_locked(e, source="savings_read")
            return

    def _maybe_trim_locked(self) -> None:
        """If the log exceeds max_bytes, keep only the most recent half. Caller
        holds self._lock. Best-effort; never raises into the caller."""
        try:
            if self._max_bytes <= 0 or os.path.getsize(self._path) <= self._max_bytes:
                return
            with open(self._path, encoding="utf-8") as f:
                lines = f.readlines()
            if len(lines) <= 1:
                return  # a single oversized record; nothing useful to trim to
            keep = lines[len(lines) // 2:]  # most recent half
            tmp = self._path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.writelines(keep)
            os.replace(tmp, self._path)
        except OSError:
            pass

    def report(self) -> dict:
        # ---- Live (non-shadow) ----
        total_live = 0
        exact_hits = 0
        semantic_hits = 0
        single_flight_hits = 0
        saved_usd = 0.0
        spent_usd = 0.0
        hit_latency_ms = 0.0
        miss_latency_ms = 0.0

        # ---- Shadow ----
        total_shadow = 0
        shadow_exact = 0
        shadow_semantic = 0
        would_have_saved_usd = 0.0

        # ---- Per-model / per-tag aggregates ----
        by_model: dict = {}
        by_tag: dict = {}

        # ---- v0.13: pricing provenance + capacity inputs ----
        # Any USD in this report that rests on the generic `_default` fallback
        # is NOT a real price for that model — count it, name the models, and
        # expose the total so it can never pass silently as customer truth.
        # Pre-v0.13 records carry no pricing_source; classify them against the
        # CURRENT pricing registry (deterministic best-effort).
        from tokeymeter.engines.economics.pricing import pricing_info as _pinfo
        _psrc_cache: dict = {}
        default_priced_calls = 0
        default_priced_usd = 0.0
        default_priced_models: set = set()
        saved_input_tokens = 0
        saved_output_tokens = 0

        def _bucket(d: dict, key: str) -> dict:
            return d.setdefault(key, {
                "calls": 0,
                "exact_hits": 0,
                "semantic_hits": 0,
                "single_flight_hits": 0,
                "shadow_exact": 0,
                "shadow_semantic": 0,
                "saved_usd": 0.0,
                "spent_usd": 0.0,
                "would_have_saved_usd": 0.0,
            })

        for rec in self._iter_records():
            cost = float(rec.get("estimated_cost", 0.0))
            latency = float(rec.get("latency_ms", 0.0))
            hit_type = rec.get("hit_type")
            is_shadow = bool(rec.get("shadow", False))
            model = rec.get("model", "_unknown")
            tag = rec.get("tag")

            # pricing provenance (v0.13): recorded at call time; legacy records
            # fall back to classification against the current registry.
            src = rec.get("pricing_source")
            if src is None:
                src = _psrc_cache.get(model)
                if src is None:
                    src = _pinfo(model)["source"]
                    _psrc_cache[model] = src
            if src == "default":
                default_priced_calls += 1
                default_priced_usd += cost
                default_priced_models.add(model)
            if not is_shadow and rec.get("hit"):
                saved_input_tokens += int(rec.get("input_tokens", 0) or 0)
                saved_output_tokens += int(rec.get("output_tokens", 0) or 0)

            bm = _bucket(by_model, model)
            bm["calls"] += 1
            bt = _bucket(by_tag, tag or "_untagged")
            bt["calls"] += 1

            if is_shadow and hit_type and hit_type.startswith("shadow_"):
                # Shadow hit: real call still happened, but we'd have saved.
                total_shadow += 1
                would_have_saved_usd += cost
                bm["would_have_saved_usd"] += cost
                bt["would_have_saved_usd"] += cost
                if hit_type == "shadow_exact":
                    shadow_exact += 1
                    bm["shadow_exact"] += 1
                    bt["shadow_exact"] += 1
                elif hit_type == "shadow_semantic":
                    shadow_semantic += 1
                    bm["shadow_semantic"] += 1
                    bt["shadow_semantic"] += 1
                # Shadow lookups always end in a real call → also count
                # the cost as "spent" since the API actually ran.
                spent_usd += cost
                miss_latency_ms += latency
                bm["spent_usd"] += cost
                bt["spent_usd"] += cost
                total_live += 1  # the real call happened
                continue

            total_live += 1
            if rec.get("hit"):
                saved_usd += cost
                hit_latency_ms += latency
                bm["saved_usd"] += cost
                bt["saved_usd"] += cost
                if hit_type == "exact":
                    exact_hits += 1
                    bm["exact_hits"] += 1
                    bt["exact_hits"] += 1
                elif hit_type == "semantic":
                    semantic_hits += 1
                    bm["semantic_hits"] += 1
                    bt["semantic_hits"] += 1
                elif hit_type == "single_flight":
                    single_flight_hits += 1
                    bm["single_flight_hits"] += 1
                    bt["single_flight_hits"] += 1
            else:
                spent_usd += cost
                miss_latency_ms += latency
                bm["spent_usd"] += cost
                bt["spent_usd"] += cost
        cache_hits = exact_hits + semantic_hits + single_flight_hits
        misses = total_live - cache_hits

        def _hit_rate(num, denom):
            return round((num / denom * 100) if denom else 0.0, 2)

        return {
            # ---- Top-level live numbers ----
            "total_calls": total_live,
            "cache_hits": cache_hits,
            "exact_hits": exact_hits,
            "semantic_hits": semantic_hits,
            "single_flight_hits": single_flight_hits,
            "cache_misses": misses,
            "hit_rate_pct": _hit_rate(cache_hits, total_live),
            "exact_hit_rate_pct": _hit_rate(exact_hits, total_live),
            "semantic_hit_rate_pct": _hit_rate(semantic_hits, total_live),
            "single_flight_hit_rate_pct": _hit_rate(single_flight_hits, total_live),
            "estimated_saved_usd": round(saved_usd, 6),
            "estimated_spent_usd": round(spent_usd, 6),
            "avg_hit_latency_ms": round(
                (hit_latency_ms / cache_hits) if cache_hits else 0.0, 3),
            "avg_miss_latency_ms": round(
                (miss_latency_ms / misses) if misses else 0.0, 3),

            # ---- v0.13: pricing provenance (anti-fabrication) ----
            # all_priced=True means every USD above rests on a registered or
            # public list price. If False, `default_priced_usd` is the exact
            # amount derived from the generic fallback — treat it as unpriced,
            # and register true rates (register_pricing / register_selfhost_pricing)
            # for the models named here.
            "pricing": {
                "all_priced": default_priced_calls == 0,
                "default_priced_calls": default_priced_calls,
                "default_priced_usd": round(default_priced_usd, 6),
                "default_priced_models": sorted(default_priced_models),
            },
            # ---- v0.13: capacity inputs (self-hosted savings unit) ----
            # Tokens NOT generated because a hit answered instead. Feed these to
            # tokeymeter.capacity_report(measured_tokens_per_second, ...) to
            # express savings in GPU-hours reclaimed on the caller's own cluster.
            "saved_input_tokens": saved_input_tokens,
            "saved_output_tokens": saved_output_tokens,

            # ---- Shadow block ----
            "shadow": {
                "total_shadow_lookups": total_shadow,
                "shadow_exact_hits": shadow_exact,
                "shadow_semantic_hits": shadow_semantic,
                "would_have_saved_usd": round(would_have_saved_usd, 6),
                "shadow_hit_rate_pct": _hit_rate(
                    shadow_exact + shadow_semantic, total_shadow),
            },

            # ---- Per-model / per-tag ----
            "by_model": {
                m: {
                    "calls": v["calls"],
                    "exact_hits": v["exact_hits"],
                    "semantic_hits": v["semantic_hits"],
                    "single_flight_hits": v["single_flight_hits"],
                    "shadow_hits": v["shadow_exact"] + v["shadow_semantic"],
                    "hit_rate_pct": _hit_rate(
                        v["exact_hits"] + v["semantic_hits"] + v["single_flight_hits"],
                        v["calls"]),
                    "estimated_saved_usd": round(v["saved_usd"], 6),
                    "estimated_spent_usd": round(v["spent_usd"], 6),
                    "would_have_saved_usd": round(v["would_have_saved_usd"], 6),
                }
                for m, v in by_model.items()
            },
            "by_tag": {
                t: {
                    "calls": v["calls"],
                    "exact_hits": v["exact_hits"],
                    "semantic_hits": v["semantic_hits"],
                    "single_flight_hits": v["single_flight_hits"],
                    "shadow_hits": v["shadow_exact"] + v["shadow_semantic"],
                    "hit_rate_pct": _hit_rate(
                        v["exact_hits"] + v["semantic_hits"] + v["single_flight_hits"],
                        v["calls"]),
                    "estimated_saved_usd": round(v["saved_usd"], 6),
                    "estimated_spent_usd": round(v["spent_usd"], 6),
                    "would_have_saved_usd": round(v["would_have_saved_usd"], 6),
                }
                for t, v in by_tag.items()
            },
        }

    def reset(self) -> None:
        try:
            with self._lock:
                self._buffer = []
                self._mem = []
                self._bytes_since_check = 0
                if os.path.exists(self._path):
                    os.remove(self._path)
        except OSError:
            pass


_tracker = SavingsTracker()


def savings_report() -> dict:
    """What the optimization layer avoided, with its own honesty block.

    Reports avoided spend by mechanism (exact cache, semantic cache,
    single-flight collapse, compression) alongside what was actually executed,
    so the two are never conflated. Mechanisms that are not yet measured report
    zero rather than an estimate, and the report states which is which — a
    saving nobody can audit is not a saving.
    """
    return _tracker.report()


def capacity_report(measured_tokens_per_second, gpu_hour_rate_usd=None) -> dict:
    """Savings expressed in GPU capacity — the honest unit for self-hosters.

    Reads the live ledger's saved-token totals (tokens NOT generated because a
    cache/single-flight hit answered instead) and converts them to GPU time
    using the caller's OWN measured throughput:

        gpu_seconds_reclaimed = saved_tokens / measured_tokens_per_second

    A USD equivalence is included only if the caller supplies their own
    gpu_hour_rate_usd. Every figure derives from measured inputs; nothing is
    assumed, and the derivation is embedded in the result.
    """
    from tokeymeter.engines.economics.pricing import capacity_reclaimed
    rep = _tracker.report()
    return capacity_reclaimed(
        saved_input_tokens=rep.get("saved_input_tokens", 0),
        saved_output_tokens=rep.get("saved_output_tokens", 0),
        measured_tokens_per_second=measured_tokens_per_second,
        gpu_hour_rate_usd=gpu_hour_rate_usd,
    )


def savings_ledger_health() -> dict:
    """Health of the local savings ledger: writable, degraded, counts, last error."""
    return _tracker.ledger_health()


def _rebind_tracker() -> None:
    """Re-resolve the ledger path after an override (set_home / set_savings_path),
    preserving the active write mode so it isn't silently reset by a path change."""
    global _tracker
    old = _tracker
    new = SavingsTracker()
    if getattr(old, "_in_memory", False):
        new.set_in_memory(True)
    elif getattr(old, "_buffered", False):
        new.set_buffered(True, old._buffer_max, old._flush_interval)
    _tracker = new


def set_home(path: Optional[str]) -> None:
    """Point all Tokeymeter local state at a new home dir (CI, containers,
    sandboxes). Takes effect immediately for the savings ledger."""
    paths.set_home(path)
    _rebind_tracker()


def set_savings_path(path: Optional[str]) -> None:
    """Point the savings ledger at a specific file. Takes effect immediately."""
    paths.set_savings_path(path)
    _rebind_tracker()


def set_buffered_savings(enabled: bool = True, buffer_size: int = 128,
                         flush_interval: float = 0.0) -> None:
    """Batch savings writes off the hot path (high-throughput / slow-disk envs).

    Records accumulate in memory and are written in batches when `buffer_size`
    lines accumulate, on `flush_interval` seconds (if > 0), on `flush_savings()`,
    and at process exit. Durable; just not synchronous per call."""
    _tracker.set_buffered(enabled, buffer_size, flush_interval)


def set_in_memory_savings(enabled: bool = True) -> None:
    """Keep savings records in RAM only — no disk I/O at all. For benchmarks and
    ephemeral/read-only environments. `report()` still works from memory."""
    _tracker.set_in_memory(enabled)


def flush_savings() -> None:
    """Persist any buffered savings records to disk now."""
    _tracker.flush()


# Durability: never lose buffered records on a clean exit.
atexit.register(lambda: _tracker.flush())


def reset_savings() -> None:
    _tracker.reset()


def _record(rec: CallRecord) -> None:
    _tracker.record(rec)
