"""
RedisStore — the shared, distributed cache backend (v0.9).

This is the credibility floor for production: without a shared backend, every
pod in a fleet has its own cache and the real-world hit rate collapses.
RedisStore lets N processes share one cache.

Design contract (enforced by tests):
  - FAIL-OPEN ALWAYS. If Redis is unreachable, slow, or returns garbage,
    get() returns None (cache miss) and set() is a silent no-op. A Redis
    outage degrades Tokeymeter to "call the real function every time" — it NEVER
    raises into the caller's code or takes down the app.
  - NATIVE TTL. The decorator wraps values in a TTL envelope; RedisStore
    reads the envelope's expiry and sets a matching Redis EXPIRE, so expired
    entries are evicted by Redis itself (defense in depth with the envelope
    check the decorator already does).
  - NAMESPACED. All keys live under a configurable prefix so Tokeymeter can share a
    Redis instance with other workloads without collision.
  - ZERO-KNOWLEDGE (optional). With a Cipher, values are encrypted at rest.
    Combined with already-hashed keys, the Redis instance holds only opaque
    hashes and ciphertext.
  - STAMPEDE PROTECTION (optional, opt-in). Distributed single-flight via a
    Redis lock: when many pods miss the same key at once, one computes and
    the rest wait for the result — preventing a thundering herd of identical
    model calls. Also fail-open: if the lock or wait misbehaves, every pod
    just computes (correct, if briefly redundant).

The store implements the same get/set/clear/__len__ interface as MemoryStore
and SQLiteStore, so it is a drop-in:

    import tokeymeter
    from tokeymeter.backends import RedisStore
    tokeymeter.set_default_store(RedisStore(url="redis://localhost:6379/0"))
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, Optional

from tokeymeter.engines.trust.cipher import Cipher, NoOpCipher

log = logging.getLogger("tokeymeter.backends.redis")


# The envelope format the decorator uses: {"__tokeymeter_v1__": [value, expires_at]}.
# We parse it best-effort to derive a native Redis TTL. If the format ever
# changes, we fail-open to "no native TTL" (the envelope check still works).
def _extract_expiry(value: Any) -> Optional[float]:
    try:
        if isinstance(value, dict) and len(value) == 1:
            (only_key, payload), = value.items()
            if (isinstance(only_key, str) and only_key.startswith("__tokeymeter_v")
                    and isinstance(payload, (list, tuple)) and len(payload) == 2):
                expires_at = payload[1]
                if expires_at is not None:
                    return float(expires_at)
    except Exception:
        pass
    return None


class RedisStore:
    """Shared, distributed, fail-open cache backed by Redis.

    Args:
        url: redis URL (e.g. "redis://localhost:6379/0"). Mutually exclusive
            with `client`.
        client: an existing redis.Redis instance (dependency injection — use
            this to share a connection pool or pass a fakeredis client in
            tests).
        namespace: key prefix for all Tokeymeter keys. Default "tokeymeter".
        cipher: optional Cipher for at-rest value encryption (zero-knowledge
            cache). Default: NoOpCipher (plaintext, an explicit choice).
        socket_timeout: per-op Redis timeout in seconds. Kept SMALL (0.5s) so
            a slow/hung Redis fails open quickly rather than stalling calls.
        default_ttl: fallback TTL (seconds) applied when a stored value has no
            envelope expiry. None = no native expiry (rely on envelope).
        single_flight_timeout: how long a non-leader waits for the leader's
            result before computing itself (seconds).
        single_flight_poll: poll interval while waiting (seconds).
        lock_ttl: max lifetime of the compute lock (seconds) — a safety net so
            a crashed leader can't block others forever.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        *,
        client: Optional[Any] = None,
        namespace: str = "tokeymeter",
        cipher: Optional[Cipher] = None,
        key_secret: Optional[bytes] = None,
        socket_timeout: float = 0.5,
        default_ttl: Optional[float] = None,
        single_flight_timeout: float = 10.0,
        single_flight_poll: float = 0.05,
        lock_ttl: float = 30.0,
    ):
        if client is not None and url is not None:
            raise ValueError("provide either url or client, not both")

        if client is not None:
            self._r = client
        else:
            try:
                import redis
            except ImportError as e:
                raise ImportError(
                    "RedisStore requires the 'redis' package. "
                    "Install with: pip install tokeymeter[scale]  (or: pip install redis)"
                ) from e
            self._r = redis.Redis.from_url(
                url or "redis://localhost:6379/0",
                socket_timeout=socket_timeout,
                socket_connect_timeout=socket_timeout,
                decode_responses=False,  # we handle bytes (for encryption)
            )

        self.namespace = namespace
        self._cipher: Cipher = cipher or NoOpCipher()
        # --- Enforce active security policy (opt-in; default permissive) ---
        try:
            from tokeymeter.engines.governance.policy import get_security_policy, SecurityPolicyError
            _pol = get_security_policy()
        except Exception:
            _pol = None
        if _pol is not None:
            if _pol.require_encryption and (cipher is None or isinstance(cipher, NoOpCipher)):
                raise SecurityPolicyError(
                    "SecurityPolicy.require_encryption is enabled: RedisStore must be "
                    "constructed with a real cipher (e.g. cipher=FernetCipher(key)). "
                    "Plaintext / NoOpCipher is refused. Tip: RedisStore.secure(...).")
            if _pol.require_keyed_cache and key_secret is None:
                raise SecurityPolicyError(
                    "SecurityPolicy.require_keyed_cache is enabled: RedisStore must be "
                    "constructed with key_secret=<bytes> so cache keys are opaque MACs.")
        # H2: optional deployment-stable secret used to HMAC cache keys so the
        # Redis keyspace holds only opaque MACs, not guessable SHA-256 digests.
        # Must be identical across pods (so the shared cache still shares); a
        # read-access attacker without it cannot confirm whether a guessable
        # prompt was cached. No secret => keys stay as-is (documented caveat).
        self._key_secret: Optional[bytes] = key_secret
        # H1: storing plaintext in a SHARED store is a real exposure (any DBA,
        # backup, or compromised neighbor can read every cached completion).
        # We still allow it (single-pod / trusted-Redis setups may accept it),
        # but it must be a VISIBLE, deliberate choice — never a silent default.
        if cipher is None:
            import warnings
            warnings.warn(
                "RedisStore: no cipher configured — cache VALUES are stored as "
                "PLAINTEXT in Redis. Anyone with Redis read access can read every "
                "cached response. Pass cipher=FernetCipher(key) for the "
                "zero-knowledge cache, or cipher=NoOpCipher() to acknowledge "
                "plaintext and silence this warning.",
                stacklevel=2,
            )
            log.warning("tokeymeter.redis: no cipher set — values stored in PLAINTEXT.")
        self._default_ttl = default_ttl
        self._sf_timeout = max(0.1, float(single_flight_timeout))
        self._sf_poll = max(0.005, float(single_flight_poll))
        self._lock_ttl = max(1.0, float(lock_ttl))

        # Health tracking: after a failure we briefly stop hammering Redis.
        self._healthy = True
        self._last_fail_ts = 0.0
        self._cooldown = 1.0  # seconds to wait before retrying after a failure

    # ----- key helpers -----

    def _kid(self, key: str) -> str:
        """Opaque key identifier. With a key_secret, HMAC the incoming cache
        key so the Redis keyspace reveals only MACs (closes the prompt-
        confirmation attack); without one, pass through unchanged."""
        if self._key_secret:
            import hashlib, hmac
            return hmac.new(self._key_secret, key.encode("utf-8"),
                            hashlib.sha256).hexdigest()
        return key

    def _vk(self, key: str) -> str:
        return f"{self.namespace}:v:{self._kid(key)}"

    def _lk(self, key: str) -> str:
        return f"{self.namespace}:lock:{self._kid(key)}"

    def _in_cooldown(self) -> bool:
        if self._healthy:
            return False
        if (time.time() - self._last_fail_ts) > self._cooldown:
            # Cooldown elapsed — allow a retry
            self._healthy = True
            return False
        return True

    def _mark_fail(self, op: str, e: Exception) -> None:
        was_healthy = self._healthy
        self._healthy = False
        self._last_fail_ts = time.time()
        log.debug("tokeymeter.redis: %s failed (fail-open): %s", op, e)
        # Surface the backend going DOWN exactly once per outage (on the
        # healthy->unhealthy edge), not on every call during the cooldown — so
        # operators see the transition without the bus being flooded.
        if was_healthy:
            self._emit_set_degraded("redis_unhealthy", e)

    @staticmethod
    def _emit_set_degraded(source: str, error: BaseException) -> None:
        """Surface a serialize/encrypt write failure as a degraded event so it is
        visible in metrics rather than masquerading as a cache miss."""
        try:
            from tokeymeter.engines.reliability.degraded import emit_degraded
            emit_degraded(source, error)
        except Exception:
            pass

    @classmethod
    def secure(cls, url=None, *, client=None, encryption_key, key_secret,
               namespace: str = "tokeymeter", **kwargs) -> "RedisStore":
        """Construct a RedisStore on the SAFE path in one call: encrypted values
        (Fernet) AND opaque HMAC'd cache keys. Both secrets are REQUIRED — Tokeymeter
        never invents keys (you own key management).

            store = RedisStore.secure(url="redis://...",
                                      encryption_key=Fernet.generate_key(),
                                      key_secret=os.environ["TOKEYMETER_KEY_SECRET"].encode())
        """
        from tokeymeter.engines.trust.cipher import FernetCipher
        return cls(url=url, client=client, namespace=namespace,
                   cipher=FernetCipher(key=encryption_key),
                   key_secret=key_secret, **kwargs)

    # ----- core get/set (fail-open) -----

    def get(self, key: str) -> Optional[Any]:
        if self._in_cooldown():
            return None
        try:
            raw = self._r.get(self._vk(key))
            if raw is None:
                self._healthy = True
                return None
            try:
                plaintext = self._cipher.decrypt(raw)
            except Exception as e:
                # Tampered / wrong key / corrupt → treat as miss (fail-open)
                log.debug("tokeymeter.redis: decrypt failed, treating as miss: %s", e)
                return None
            value = json.loads(plaintext.decode("utf-8"))
            self._healthy = True
            return value
        except Exception as e:
            self._mark_fail("get", e)
            return None

    def set(self, key: str, value: Any) -> None:
        if self._in_cooldown():
            return
        try:
            payload = json.dumps(value, default=str).encode("utf-8")
        except (TypeError, ValueError) as e:
            self._emit_set_degraded("redis_serialize", e)
            return  # unserializable → surfaced as degraded, not a phantom miss
        try:
            ciphertext = self._cipher.encrypt(payload)
        except Exception as e:
            log.debug("tokeymeter.redis: encrypt failed, skipping set: %s", e)
            self._emit_set_degraded("redis_encrypt", e)
            return
        # Derive native TTL from the envelope expiry, if any.
        ttl_seconds: Optional[float] = None
        expires_at = _extract_expiry(value)
        if expires_at is not None:
            ttl_seconds = max(1.0, expires_at - time.time())
        elif self._default_ttl is not None:
            ttl_seconds = self._default_ttl
        try:
            if ttl_seconds is not None:
                self._r.set(self._vk(key), ciphertext, ex=int(ttl_seconds))
            else:
                self._r.set(self._vk(key), ciphertext)
            self._healthy = True
        except Exception as e:
            # Track health/cooldown AND surface it as a degraded event, so a real
            # write failure is as visible as the SQLite equivalent (sqlite_write)
            # instead of only flipping the backend unhealthy.
            self._mark_fail("set", e)
            self._emit_set_degraded("redis_write", e)

    def clear(self) -> None:
        """Delete all Tokeymeter keys in this namespace. Fail-open."""
        try:
            pattern = f"{self.namespace}:*"
            cursor = 0
            while True:
                cursor, keys = self._r.scan(cursor=cursor, match=pattern, count=500)
                if keys:
                    self._r.delete(*keys)
                if cursor == 0:
                    break
        except Exception as e:
            log.debug("tokeymeter.redis: clear failed (fail-open): %s", e)

    def __len__(self) -> int:
        """Count value keys in this namespace. Fail-open → 0."""
        try:
            pattern = f"{self.namespace}:v:*"
            cursor = 0
            n = 0
            while True:
                cursor, keys = self._r.scan(cursor=cursor, match=pattern, count=500)
                n += len(keys)
                if cursor == 0:
                    break
            return n
        except Exception as e:
            log.debug("tokeymeter.redis: len failed (fail-open): %s", e)
            return 0

    # ----- health -----

    def ping(self) -> bool:
        """True if Redis is reachable. Never raises."""
        try:
            return bool(self._r.ping())
        except Exception:
            return False

    # ----- distributed single-flight (opt-in, duck-typed by the decorator) -----
    #
    # The decorator calls these IF the store provides them. All are fail-open:
    # any error results in "compute it yourself", which is correct (just
    # potentially redundant) rather than wrong.

    def acquire_compute_lock(self, key: str) -> "Optional[str]":
        """Try to become the leader for computing `key`.

        Returns a unique token if acquired, else None. Uses SET NX PX so the
        lock auto-expires (lock_ttl) even if the leader crashes.
        """
        if self._in_cooldown():
            return None
        token = uuid.uuid4().hex
        try:
            ok = self._r.set(
                self._lk(key), token, nx=True, px=int(self._lock_ttl * 1000)
            )
            return token if ok else None
        except Exception as e:
            self._mark_fail("acquire_lock", e)
            return None

    def release_compute_lock(self, key: str, token: str) -> None:
        """Release the lock IF we still own it (compare-and-delete).

        Prefers an atomic Lua compare-and-delete on real Redis so we never
        delete a lock a later leader took over after our TTL expired. Where
        server-side scripting is unavailable (e.g. fakeredis, or a managed
        Redis with eval disabled), falls back to a get-then-conditional-delete.
        The fallback has a tiny benign race (we could delete a successor's lock
        only if our process stalls for the full lock TTL between get and del);
        the worst case is one redundant computation, never incorrect data.
        Fail-open throughout.
        """
        lk = self._lk(key)
        _RELEASE = (
            "if redis.call('get', KEYS[1]) == ARGV[1] then "
            "return redis.call('del', KEYS[1]) else return 0 end"
        )
        # Preferred path: atomic server-side compare-and-delete.
        try:
            self._r.eval(_RELEASE, 1, lk, token)
            return
        except Exception:
            pass  # eval unsupported or transient → portable fallback below
        # Portable fallback: compare-and-delete in the client.
        try:
            current = self._r.get(lk)
            if current is not None:
                cur = current.decode() if isinstance(current, bytes) else current
                if cur == token:
                    self._r.delete(lk)
        except Exception as e:
            log.debug("tokeymeter.redis: release fallback failed (fail-open): %s", e)

    def wait_for_result(self, key: str, timeout: Optional[float] = None) -> Optional[Any]:
        """Poll for another pod's result up to `timeout`. None if it never came.

        Used by non-leaders: while the leader computes, we poll the value key.
        Fail-open: on timeout the caller computes itself.
        """
        deadline = time.time() + (timeout if timeout is not None else self._sf_timeout)
        while time.time() < deadline:
            val = self.get(key)
            if val is not None:
                return val
            time.sleep(self._sf_poll)
        return None
