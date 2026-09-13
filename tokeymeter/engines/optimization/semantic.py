"""
Semantic prompt caching (Method 1 from the optimization roadmap).

The biggest single cost lever in the library. Where exact-match caching
catches identical prompts, semantic caching catches prompts that mean
the same thing said differently.

Two backends, auto-selected at runtime:
  - **sqlite-vec backend**: indexed nearest-neighbor lookup via the
    vec0 virtual table. Scales to millions of entries. Used when
    `pip install tokeymeter[scale]` is installed AND the SQLite build
    supports extension loading.
  - **Linear-scan backend**: load all embeddings as a numpy matrix,
    matmul against the query. Fine to ~10k entries (a few ms scan time).
    Always available.

Both implement the same interface and the same fail-open semantics:
on any internal error, return None / silently skip writes. The decorator
treats None as a cache miss and falls through to the real API call.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

log = logging.getLogger("tokeymeter.semantic")

# ---------- Lazy dependency imports ----------

_numpy = None
_st_model_loader = None
_sqlite_vec = None
_import_attempted = False
_import_lock = threading.Lock()


def _try_import() -> None:
    """Idempotent best-effort import of numpy / sentence-transformers / sqlite-vec."""
    global _numpy, _st_model_loader, _sqlite_vec, _import_attempted
    with _import_lock:
        if _import_attempted:
            return
        _import_attempted = True
        try:
            import numpy as np
            _numpy = np
        except ImportError:
            log.debug("tokeymeter.semantic: numpy not installed")
        try:
            from sentence_transformers import SentenceTransformer
            _st_model_loader = SentenceTransformer
        except ImportError:
            log.debug("tokeymeter.semantic: sentence-transformers not installed")
        try:
            import sqlite_vec
            _sqlite_vec = sqlite_vec
        except ImportError:
            log.debug("tokeymeter.semantic: sqlite-vec not installed; linear scan only")


def is_available() -> bool:
    """True if numpy + sentence-transformers are available."""
    _try_import()
    return _numpy is not None and _st_model_loader is not None


def is_vec_index_available() -> bool:
    """True if sqlite-vec can accelerate lookups."""
    _try_import()
    if _sqlite_vec is None or _numpy is None:
        return False
    # Verify the SQLite build supports extension loading
    try:
        conn = sqlite3.connect(":memory:")
        try:
            conn.enable_load_extension(True)
        except (AttributeError, sqlite3.OperationalError):
            return False
        finally:
            conn.close()
        return True
    except Exception:
        return False


# ---------- Model cache (one process-global instance per model name) ----------

_model_cache: dict = {}
_model_cache_lock = threading.Lock()


def default_encoder(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> Callable[[str], Any]:
    if not is_available():
        raise RuntimeError(
            "Semantic caching requires extras. Install with: "
            "pip install tokeymeter[semantic]"
        )

    def _encode(text: str):
        with _model_cache_lock:
            model = _model_cache.get(model_name)
            if model is None:
                model = _st_model_loader(model_name)
                _model_cache[model_name] = model
        return model.encode(text, convert_to_numpy=True, normalize_embeddings=True).astype("float32")

    return _encode


# ============================================================
#                          MAIN CLASS
# ============================================================

class _NonSerializable(Exception):
    """Signals a value that cannot be losslessly JSON-encoded, so we keep it
    live in-process instead of stringifying it (which would corrupt SDK
    response objects on a cache hit)."""


def _json_default_strict(o):
    """JSON default that REFUSES arbitrary objects instead of stringifying them.

    The old code used json.dumps(default=str), which silently turned a
    non-serializable response (an OpenAI ChatCompletion) into its string form —
    so a semantic hit returned a string, not the object, and resp.choices failed.
    This default allows only safe coercions and raises otherwise, so the caller
    can keep the live object instead.
    """
    import datetime
    if isinstance(o, (datetime.date, datetime.datetime)):
        return o.isoformat()
    raise _NonSerializable(type(o).__name__)


class SemanticCache:
    """Embedding-based prompt cache.

    Picks the fastest backend automatically:
      - sqlite-vec (vec0) if available and SQLite supports extensions
      - Linear numpy scan otherwise

    Both backends are fail-open: any internal error returns None on
    lookup and silently skips writes.
    """

    def __init__(
        self,
        path: Optional[str] = None,
        threshold: float = 0.92,
        encoder: Optional[Callable[[str], Any]] = None,
        max_entries: int = 10_000,
        use_vec_index: Optional[bool] = None,
        dim: int = 384,
        verifier: Optional[object] = None,
        monitor: Optional[object] = None,
    ):
        """
        Args:
            path: SQLite file location.
            threshold: Cosine similarity cutoff. Range [-1, 1]. Higher =
                fewer false matches but lower hit rate. 0.92 default.
            encoder: Function (text -> normalized float32 vector). If None,
                uses the default sentence-transformers encoder.
            max_entries: LRU cap.
            use_vec_index: True = require sqlite-vec, False = forbid,
                None = auto-detect.
            dim: Embedding dimension. Default 384 (MiniLM). Only matters
                when sqlite-vec is used (its vec0 schema is fixed-dim).
        """
        from tokeymeter import paths as _paths
        self._path = os.path.expanduser(
            path if path is not None else _paths.state_path("semantic.db"))
        # :memory: (and shared-cache memory URIs) have no filesystem parent and
        # lose all data across separate connections — so we both skip the mkdir
        # and hold ONE persistent connection for the object's lifetime. File-backed
        # paths keep the original per-call connect behavior.
        self._is_memory = (self._path == ":memory:" or "mode=memory" in self._path)
        if not self._is_memory:
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._shared_conn = None
        # In-process side-table holding LIVE response objects that are not
        # JSON-serializable (e.g. SDK ChatCompletion objects). The DB stores a
        # reference token; this maps the token back to the live object so a
        # semantic hit returns the real object, not a stringified form.
        self._live_objects: dict = {}
        self._live_max = 10_000
        self._threshold = float(threshold)
        self._max = max_entries
        self._dim = dim
        self._lock = threading.Lock()
        # Stage-2 cross-encoder verifier (optional). When set, a Stage-1 cosine
        # candidate is only served if the verifier confirms the two prompts are
        # genuinely the same question — eliminating near-miss false hits. When
        # None, the cache behaves as Stage-1-only (backward compatible).
        self._verifier = verifier
        # Eval loop (optional). When set, every served semantic hit is recorded
        # (content-blind) for false-positive monitoring. Never affects what is
        # served — pure observation (plus opt-in auto-tighten inside the loop).
        self._monitor = monitor

        _try_import()

        if encoder is not None:
            self._encoder = encoder
        else:
            try:
                self._encoder = default_encoder()
            except RuntimeError:
                self._encoder = None

        # Decide backend
        if use_vec_index is None:
            self._use_vec = is_vec_index_available()
        elif use_vec_index and not is_vec_index_available():
            log.debug("tokeymeter.semantic: vec index requested but unavailable")
            self._use_vec = False
        else:
            self._use_vec = bool(use_vec_index)

        # True once a vec row may exist on disk. It STAYS true even after the vec
        # backend self-disables, so LRU eviction keeps cleaning the matching
        # semantic_vec rows instead of orphaning them. Seeded from the initial
        # backend state: if vec is active at construction, the vec table exists and
        # may already hold rows from a prior run.
        self._vec_ever_used = self._use_vec

        self._init_db()

    @property
    def is_functional(self) -> bool:
        return self._encoder is not None and _numpy is not None

    @property
    def backend(self) -> str:
        """Human-readable backend name. Useful for logging / debugging."""
        return "sqlite-vec" if self._use_vec else "linear-scan"

    # ---------- Schema & connection ----------

    def _connect(self, force_vec: bool = False) -> sqlite3.Connection:
        if getattr(self, "_is_memory", False):
            # Reuse one connection for in-memory DBs; a fresh connect would get a
            # separate empty database and silently lose all stored entries.
            if self._shared_conn is None:
                self._shared_conn = sqlite3.connect(
                    self._path, timeout=2.0, check_same_thread=False)
            conn = self._shared_conn
        else:
            conn = sqlite3.connect(self._path, timeout=2.0)
        # Load the vec0 extension when vec mode is active, OR when a caller needs
        # it for table MAINTENANCE (force_vec) even though _use_vec has since
        # self-disabled — otherwise a DELETE FROM semantic_vec would fail with
        # "no such module: vec0" and silently leave orphaned vector rows behind.
        if (self._use_vec or force_vec) and _sqlite_vec is not None:
            try:
                conn.enable_load_extension(True)
                _sqlite_vec.load(conn)
                conn.enable_load_extension(False)
            except (AttributeError, sqlite3.OperationalError) as e:
                log.debug("tokeymeter.semantic: failed to load vec extension: %s", e)
                if not force_vec:
                    # Maintenance loads must not mutate operational state.
                    self._use_vec = False
                    # Runtime self-heal (vec was active, load now failing): make
                    # the degradation visible — fires once on the True->False edge.
                    self._emit_degraded("semantic_vec_load_failed", e)
        return conn

    @staticmethod
    def _emit_degraded(source: str, error: BaseException) -> None:
        try:
            from tokeymeter.engines.reliability.degraded import emit_degraded
            emit_degraded(source, error)
        except Exception:
            pass

    def _init_db(self) -> None:
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS semantic_cache (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        prompt TEXT NOT NULL,
                        embedding BLOB NOT NULL,
                        response TEXT NOT NULL,
                        created_at REAL NOT NULL
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_sem_created "
                    "ON semantic_cache(created_at)"
                )
                conn.execute("PRAGMA journal_mode=WAL")

                if self._use_vec:
                    conn.execute(
                        f"""
                        CREATE VIRTUAL TABLE IF NOT EXISTS semantic_vec
                        USING vec0(embedding float[{self._dim}])
                        """
                    )
                conn.commit()
        except (sqlite3.Error, OSError) as e:
            log.debug("tokeymeter.semantic: db init failed: %s", e)

    # ---------- Public API: lookup ----------

    def encode(self, prompt: str) -> Optional[Any]:
        """Compute and return the embedding for a prompt.

        Callers can reuse the returned embedding across lookup and store
        to avoid re-encoding on the miss path (encoding is the slow step,
        ~5-10 ms with sentence-transformers).
        """
        if not self.is_functional or not isinstance(prompt, str) or not prompt:
            return None
        try:
            emb = self._encoder(prompt)
            if emb is None:
                return None
            # Normalize any encoder output (list/tuple/ndarray) to a contiguous
            # float32 ndarray. Without this, a custom encoder returning a plain
            # Python list stores as float32 bytes but queries as a list, so the
            # stored-vs-query dtypes diverge and true matches silently miss.
            if _numpy is not None:
                emb = _numpy.ascontiguousarray(emb, dtype="float32")
            return emb
        except Exception as e:
            log.debug("tokeymeter.semantic: encode failed: %s", e)
            return None

    def lookup(self, prompt: str) -> Optional[Any]:
        """Encode and look up. Convenience wrapper around encode + lookup_by_embedding."""
        emb = self.encode(prompt)
        if emb is None:
            return None
        return self.lookup_by_embedding(emb, query_prompt=prompt)

    def lookup_by_embedding(self, query_emb, query_prompt=None) -> Optional[Any]:
        """Look up using a pre-computed embedding. Saves one encode per call.

        `query_prompt` (the original text) is passed through so the optional
        Stage-2 cross-encoder verifier can confirm a candidate is a true
        equivalent before serving. If omitted, Stage 2 is skipped (Stage-1-only).
        """
        if _numpy is None or query_emb is None:
            return None
        try:
            if self._use_vec:
                return self._lookup_vec(query_emb, query_prompt)
            return self._lookup_linear(query_emb, query_prompt)
        except Exception as e:
            log.debug("tokeymeter.semantic: lookup failed: %s", e)
            return None

    def _lookup_linear(self, query_emb, query_prompt=None) -> Optional[Any]:
        """Numpy matmul over all stored embeddings. Fine to ~10k entries.

        Stage 1: rank all stored prompts by cosine similarity to the query.
        Stage 2 (if a verifier is configured AND we have the query text): take
        the top-K above threshold and let the cross-encoder confirm a true
        semantic equivalent before serving — eliminating near-miss false hits.
        Without a verifier, behaves exactly as before (serve the top match).
        """
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT embedding, response, prompt FROM semantic_cache"
                ).fetchall()
        except (sqlite3.Error, OSError):
            return None
        if not rows:
            return None

        np = _numpy
        try:
            qdim = int(np.asarray(query_emb).shape[-1])
            keep, kept_rows = [], []
            for blob, resp, pr in rows:
                vec = np.frombuffer(blob, dtype=np.float32)
                if vec.shape[-1] == qdim:      # only same-dim embeddings compare
                    keep.append(vec)
                    kept_rows.append((blob, resp, pr))
            if not keep:
                return None
            embs = np.stack(keep)
            sims = embs @ np.asarray(query_emb, dtype=np.float32)
            rows = kept_rows
        except Exception:
            return None

        return self._select_with_optional_verify(sims, rows, query_prompt)

    def _select_with_optional_verify(self, sims, rows, query_prompt):
        """Shared Stage-1 -> Stage-2 selection used by both lookup backends.

        `rows` is a list of (embedding_blob, response, prompt). `sims` is the
        parallel array of cosine similarities. Returns the resolved response of
        the chosen match, or None for a (verified) miss. On a served hit, notifies
        the eval loop (content-blind) for false-positive monitoring.
        """
        np = _numpy
        # Stage 1: everything at/above the cosine threshold, best-first.
        order = np.argsort(sims)[::-1]
        candidate_idxs = [int(i) for i in order if float(sims[int(i)]) >= self._threshold]
        if not candidate_idxs:
            return None

        verifier = self._verifier
        use_verifier = (verifier is not None and query_prompt is not None
                        and getattr(verifier, "available", False))

        matched_idx = None
        verify_score = None
        if not use_verifier:
            # Stage-1-only (backward compatible): serve the top cosine match.
            matched_idx = candidate_idxs[0]
            raw_response = rows[matched_idx][1]
        else:
            # Stage 2: hand the top-K candidates (with prompt text) to the
            # cross-encoder, which confirms a genuine equivalent or rejects all.
            topk = candidate_idxs[: getattr(verifier, "_max_pairs", 5)]
            candidates = [(rows[i][2] or "", i) for i in topk]  # (prompt, row_index)
            picked = self._verify_pick(verifier, query_prompt, candidates)
            if picked is None:
                return None  # Stage 1 matched, but Stage 2 found no true equivalent
            matched_idx, verify_score = picked
            raw_response = rows[matched_idx][1]

        response = self._decode_response(raw_response)
        # notify the eval loop on a served hit (content-blind; never blocks serving)
        if response is not None and self._monitor is not None and query_prompt is not None:
            try:
                self._monitor.record_hit(
                    query=query_prompt,
                    matched_prompt=rows[matched_idx][2] or "",
                    response=response,
                    similarity=float(sims[matched_idx]),
                    verify_score=verify_score,
                )
            except Exception:
                pass  # monitoring must never break a cache hit
        return response

    def _verify_pick(self, verifier, query_prompt, candidates):
        """Run the verifier over (prompt, row_index) candidates; return
        (winning_row_index, score) or None. Mirrors best_verified but keeps the
        row index so the cache can report the matched prompt to the monitor."""
        # apply the lexical guard + cross-encoder via the verifier's own logic,
        # but we need the index, so we score here using its public surface.
        guard = getattr(verifier, "_use_lexical_guard", False)
        try:
            from tokeymeter.engines.optimization.semantic_verify import lexical_veto
        except Exception:
            lexical_veto = None
        pool = []
        for prompt, idx in candidates:
            if guard and lexical_veto is not None and lexical_veto(query_prompt, prompt):
                continue
            pool.append((prompt, idx))
        if not pool:
            return None
        try:
            scores = verifier._model.predict([(query_prompt, p) for p, _ in pool])
        except Exception:
            return None
        thr = getattr(verifier, "_accept_threshold", 0.5)
        best = None
        for (_, idx), sc in zip(pool, scores):
            sc = float(sc)
            if sc >= thr and (best is None or sc > best[1]):
                best = (idx, sc)
        return best

    def _decode_response(self, raw):
        try:
            return self._resolve_live(json.loads(raw))
        except (json.JSONDecodeError, TypeError):
            return None

    def _lookup_vec(self, query_emb, query_prompt=None) -> Optional[Any]:
        """vec0 indexed nearest-neighbor + Stage-1 cosine + optional Stage-2 verify.

        Fetch the top-K candidates from the vec0 index, then apply the same
        Stage-1 (exact cosine threshold) -> Stage-2 (cross-encoder verify)
        selection as the linear path.
        """
        try:
            query_blob = _sqlite_vec.serialize_float32(query_emb.tolist())
        except Exception:
            return None

        try:
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT semantic_cache.embedding, semantic_cache.response,
                           semantic_cache.prompt
                    FROM semantic_vec
                    JOIN semantic_cache ON semantic_cache.id = semantic_vec.rowid
                    WHERE semantic_vec.embedding MATCH ? AND k = 5
                    ORDER BY semantic_vec.distance
                    """,
                    (query_blob,),
                ).fetchall()
        except (sqlite3.Error, OSError):
            return None
        if not rows:
            return None

        np = _numpy
        try:
            sims = np.array([
                float(np.dot(query_emb, np.frombuffer(emb_blob, dtype=np.float32)))
                for emb_blob, _, _ in rows
            ])
        except Exception:
            return None

        return self._select_with_optional_verify(sims, rows, query_prompt)

    def _resolve_live(self, value):
        """If the stored value is a reference token to a live in-process object,
        return that object; otherwise return the value as-is. This is what makes
        a semantic hit return the original SDK response object rather than a
        stringified form."""
        if isinstance(value, str) and value.startswith("__live__:"):
            obj = self._live_objects.get(value)
            return obj if obj is not None else None
        return value

    def _evict_live_if_needed(self):
        """Bound the live-object side-table (simple FIFO) so it can't grow
        without limit in a long-running process."""
        if len(self._live_objects) > self._live_max:
            # drop the oldest ~10%
            n = max(1, self._live_max // 10)
            for k in list(self._live_objects.keys())[:n]:
                self._live_objects.pop(k, None)

    # ---------- Public API: store ----------

    def store(self, prompt: str, response: Any) -> None:
        """Encode and store. Convenience wrapper around encode + store_by_embedding."""
        emb = self.encode(prompt)
        if emb is None:
            return
        self.store_by_embedding(prompt, emb, response)

    def store_by_embedding(self, prompt: str, emb, response: Any) -> None:
        """Store with a pre-computed embedding. Saves one encode per call.

        Object fidelity: SDK response objects (e.g. an OpenAI ChatCompletion) are
        NOT natively JSON-serializable, and json.dumps(default=str) would silently
        turn them into a string — so a later semantic hit would return a string,
        not the object, breaking attribute access (resp.choices). To prevent that,
        we keep the LIVE object in an in-process side-table keyed by a content hash,
        and store that hash in the DB. On lookup we return the live object when the
        side-table still has it (same process), falling back to the JSON form
        otherwise. Plain JSON-serializable values round-trip through the DB as
        before, so cross-process persistence is unaffected for those.
        """
        if _numpy is None or emb is None or not isinstance(prompt, str) or not prompt:
            return

        # Decide how to persist the response value.
        live_token = None
        try:
            payload = json.dumps(response, default=_json_default_strict)
        except (TypeError, ValueError, _NonSerializable):
            # Not natively serializable (e.g. an SDK object): keep it live in-process
            # and store a reference token in the DB instead of a lossy string.
            live_token = f"__live__:{id(response)}:{time.time()}"
            self._live_objects[live_token] = response
            payload = json.dumps(live_token)
            self._evict_live_if_needed()

        try:
            blob = emb.astype("float32").tobytes()
        except Exception:
            return

        try:
            with self._lock:
                # force_vec when vec rows may exist (even if vec mode has since
                # self-disabled) so _evict_if_needed can clean semantic_vec rows
                # rather than orphaning them.
                with self._connect(force_vec=self._vec_ever_used) as conn:
                    cur = conn.execute(
                        "INSERT INTO semantic_cache(prompt, embedding, response, created_at) "
                        "VALUES (?, ?, ?, ?)",
                        (prompt[:2000], blob, payload, time.time()),
                    )
                    rowid = cur.lastrowid
                    if self._use_vec and rowid is not None:
                        try:
                            vec_blob = _sqlite_vec.serialize_float32(emb.tolist())
                            conn.execute(
                                "INSERT INTO semantic_vec(rowid, embedding) VALUES (?, ?)",
                                (rowid, vec_blob),
                            )
                        except Exception as e:
                            # Most common cause: embedding dim doesn't match schema.
                            # Self-heal: disable the vec backend for this instance.
                            # The main semantic_cache table still has the data, so
                            # the linear-scan fallback will work for all future lookups.
                            log.warning(
                                "tokeymeter.semantic: vec0 insert failed (%s); "
                                "falling back to linear scan for this cache instance",
                                e,
                            )
                            self._use_vec = False
                            self._emit_degraded("semantic_vec_disabled", e)
                    # LRU eviction
                    self._evict_if_needed(conn)
                    conn.commit()
        except (sqlite3.Error, OSError) as e:
            log.debug("tokeymeter.semantic: db write failed: %s", e)

    def _evict_if_needed(self, conn: sqlite3.Connection) -> None:
        """Evict oldest entries beyond max_entries."""
        try:
            cnt = conn.execute("SELECT COUNT(*) FROM semantic_cache").fetchone()[0]
            if cnt <= self._max:
                return
            # Warn ONCE: silent eviction quietly degrades hit rate. "Scales to
            # millions" requires raising max_entries; the default cap is 10k.
            if not getattr(self, "_evict_warned", False):
                self._evict_warned = True
                import warnings
                warnings.warn(
                    f"SemanticCache reached max_entries={self._max} and is now "
                    f"evicting oldest entries (LRU). Hit rate will degrade as "
                    f"entries are dropped. To scale further (the point of the "
                    f"sqlite-vec backend), construct with a larger max_entries.",
                    stacklevel=2,
                )
                log.warning("tokeymeter.semantic: cache full at %d entries; evicting (LRU).",
                            self._max)
            to_drop = cnt - self._max
            rows = conn.execute(
                "SELECT id FROM semantic_cache ORDER BY created_at ASC LIMIT ?",
                (to_drop,),
            ).fetchall()
            ids = [r[0] for r in rows]
            if not ids:
                return
            qmarks = ",".join("?" for _ in ids)
            conn.execute(f"DELETE FROM semantic_cache WHERE id IN ({qmarks})", ids)
            # Clean the matching vec rows whenever they MAY exist — including after
            # the vec backend self-disabled (otherwise LRU eviction orphans them).
            if self._use_vec or self._vec_ever_used:
                try:
                    conn.execute(f"DELETE FROM semantic_vec WHERE rowid IN ({qmarks})", ids)
                except sqlite3.OperationalError:
                    pass  # vec table/module unavailable on this conn — expected
                except Exception as e:
                    self._emit_degraded("semantic.evict_vec", e)
        except sqlite3.Error as e:
            log.debug("tokeymeter.semantic: eviction failed: %s", e)

    # ---------- Maintenance ----------

    def clear(self) -> None:
        try:
            # force_vec=True so the vec0 module is loaded for maintenance even if
            # vec mode self-disabled at runtime — otherwise previously-inserted
            # vector rows would be orphaned.
            with self._connect(force_vec=True) as conn:
                conn.execute("DELETE FROM semantic_cache")
                try:
                    conn.execute("DELETE FROM semantic_vec")
                except sqlite3.OperationalError:
                    pass  # vec table was never created (vec unavailable) — expected
                except Exception as e:
                    # vec table exists but the delete failed: rows may now be
                    # orphaned. Don't crash maintenance, but make it VISIBLE.
                    self._emit_degraded("semantic.clear_vec", e)
                conn.commit()
        except (sqlite3.Error, OSError) as e:
            self._emit_degraded("semantic.clear", e)

    def __len__(self) -> int:
        try:
            with self._connect() as conn:
                row = conn.execute("SELECT COUNT(*) FROM semantic_cache").fetchone()
                return int(row[0]) if row else 0
        except (sqlite3.Error, OSError):
            return 0
