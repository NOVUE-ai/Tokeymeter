"""Provider error taxonomy (EXEC-4, W2) — typed, SDK-agnostic, content-blind.

Maps any provider/SDK exception into five stable types the Reliability
engine can route on:

    RateLimited   retryable, may carry retry_after (REL-3b seam)
    AuthError     NOT retryable — retrying auth failures is a waste and a log-spam
    TransientError retryable — timeouts, connection resets, 408s
    MalformedRequest NOT retryable — the request itself is wrong; retrying can't fix it
    ProviderDown  retryable via fallback — 5xx / service unavailable

Classification is by exception-class NAME walk over the MRO plus an HTTP
status-code fallback, so no provider SDK is ever imported here (stdlib-only
core law holds). Unknown exceptions default to TransientError(retryable) —
the safe direction for resilience, pinned by test.

L4: typed errors carry provider, status, and the ORIGINAL EXCEPTION CLASS
NAME only — never request or response payload text.
"""
from __future__ import annotations

from typing import Any, Optional


class ProviderError(Exception):
    retryable: bool = True

    def __init__(self, provider: str, original: BaseException,
                 status: Optional[int] = None,
                 retry_after: Optional[float] = None) -> None:
        self.provider = provider
        self.status = status
        self.retry_after = retry_after
        self.original_type = type(original).__name__
        super().__init__(
            f"{type(self).__name__}(provider={provider}, "
            f"status={status}, from={self.original_type})")
        self.__cause__ = original


class RateLimited(ProviderError):
    retryable = True


class AuthError(ProviderError):
    retryable = False


class TransientError(ProviderError):
    retryable = True


class MalformedRequest(ProviderError):
    retryable = False


class ProviderDown(ProviderError):
    retryable = True


_NAME_MAP = {
    # OpenAI + Anthropic SDK exception class names (string-matched, no import)
    "RateLimitError": RateLimited,
    "AuthenticationError": AuthError,
    "PermissionDeniedError": AuthError,
    "APITimeoutError": TransientError,
    "APIConnectionError": TransientError,
    "Timeout": TransientError,
    "TimeoutError": TransientError,
    "ConnectionError": TransientError,
    "BadRequestError": MalformedRequest,
    "UnprocessableEntityError": MalformedRequest,
    "InvalidRequestError": MalformedRequest,
    "InternalServerError": ProviderDown,
    "ServiceUnavailableError": ProviderDown,
}

_STATUS_MAP = [
    ((429,), RateLimited),
    ((401, 403), AuthError),
    ((408,), TransientError),
    ((400, 404, 422), MalformedRequest),
]


def _status_of(exc: BaseException) -> Optional[int]:
    for attr in ("status_code", "status", "http_status", "code"):
        v = getattr(exc, attr, None)
        if isinstance(v, int):
            return v
        resp = getattr(exc, "response", None)
        v2 = getattr(resp, "status_code", None)
        if isinstance(v2, int):
            return v2
    return None


def _retry_after_of(exc: BaseException) -> Optional[float]:
    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", None) or getattr(exc, "headers", None)
    try:
        if headers and "Retry-After" in headers:
            return float(headers["Retry-After"])
    except (TypeError, ValueError):
        pass
    return None


def classify(exc: BaseException, provider: str) -> ProviderError:
    """Map any exception to a typed ProviderError. Never raises itself."""
    if isinstance(exc, ProviderError):
        return exc
    status = _status_of(exc)
    retry_after = _retry_after_of(exc)
    for klass in type(exc).__mro__:
        mapped = _NAME_MAP.get(klass.__name__)
        if mapped:
            return mapped(provider, exc, status=status,
                          retry_after=retry_after)
    if status is not None:
        for codes, mapped in _STATUS_MAP:
            if status in codes:
                return mapped(provider, exc, status=status,
                              retry_after=retry_after)
        if status >= 500:
            return ProviderDown(provider, exc, status=status,
                                retry_after=retry_after)
    return TransientError(provider, exc, status=status,
                          retry_after=retry_after)
