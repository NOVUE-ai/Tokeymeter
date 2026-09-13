"""Admin helpers exposed on the public API."""
from tokeymeter.decorator import _get_default_store, _get_default_semantic_cache


def clear_cache() -> None:
    """Wipe both exact and semantic caches. Useful for tests, demos, force-rebuild."""
    store = _get_default_store()
    if hasattr(store, "clear"):
        try:
            store.clear()
        except Exception:
            pass

    sem = _get_default_semantic_cache()
    if sem is not None and hasattr(sem, "clear"):
        try:
            sem.clear()
        except Exception:
            pass
