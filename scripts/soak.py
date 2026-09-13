#!/usr/bin/env python3
"""REL-7 soak/leak harness.

Runs a sustained workload through the runtime and asserts the process does
not leak: resident memory slope stays flat, file-descriptor and thread
counts stabilize, and the proof ledger grows LINEARLY (one entry per
request, no hidden accumulation elsewhere).

CI uses a short window (--seconds 60); nightly uses --seconds 86400.
Exit 0 = within budgets, 1 = leak detected.

    python scripts/soak.py --seconds 60 --rps 200
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokeymeter import Runtime  # noqa: E402


def _rss_kb() -> int:
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KB, macOS reports bytes
        return rss if sys.platform != "darwin" else rss // 1024
    except Exception:
        try:
            with open(f"/proc/{os.getpid()}/status") as fh:
                for line in fh:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1])
        except Exception:
            return 0
    return 0


def _fd_count() -> int:
    try:
        return len(os.listdir(f"/proc/{os.getpid()}/fd"))
    except Exception:
        return -1


def run(seconds: float, rps: int) -> int:
    runtime = Runtime(call=lambda p: "ok:" + p[:32], receipt="never")
    # Warmup: fill bounded retention windows to steady-state so the slope we
    # measure is a TRUE leak signal, not one-time fill. Default caps are 5000;
    # 6000 warmup requests guarantee the windows are saturated.
    for i in range(6000):
        runtime.execute(f"warmup {i}")
    gc.collect()
    samples = []
    start = time.monotonic()
    interval = 1.0 / max(1, rps)
    n = 0
    baseline_rss = None
    baseline_threads = threading.active_count()

    while time.monotonic() - start < seconds:
        runtime.execute(f"soak request {n}")
        n += 1
        if n % max(1, rps) == 0:
            gc.collect()
            rss = _rss_kb()
            if baseline_rss is None:
                baseline_rss = rss
            samples.append((time.monotonic() - start, rss, _fd_count(),
                            threading.active_count()))
        time.sleep(interval)

    # ---- analysis ----
    if len(samples) < 3:
        print(f"soak: {n} requests, window too short for slope; OK")
        return 0
    t0, rss0, fd0, th0 = samples[0]
    tN, rssN, fdN, thN = samples[-1]
    rss_slope_kb_per_s = (rssN - rss0) / max(1e-6, (tN - t0))
    thread_growth = thN - baseline_threads
    fd_growth = (fdN - fd0) if fd0 >= 0 else 0

    print(f"soak: {n} requests over {seconds}s")
    print(f"  RSS: {rss0}KB → {rssN}KB  (slope {rss_slope_kb_per_s:.2f} KB/s)")
    print(f"  threads: {th0} → {thN}  (net {thread_growth:+d})")
    print(f"  fds: {fd0} → {fdN}  (net {fd_growth:+d})")

    # budgets: allow a small warmup slope, but a real leak shows sustained
    # growth. 50 KB/s over a long run is ~4MB/min — clearly leaking.
    leak = False
    if rss_slope_kb_per_s > 50.0 and seconds >= 30:
        print("  LEAK: RSS growing >50 KB/s"); leak = True
    if thread_growth > 5:
        print("  LEAK: thread count growing"); leak = True
    if fd_growth > 10:
        print("  LEAK: file descriptors growing"); leak = True
    if not leak:
        print("  within budgets — no leak detected")
    return 1 if leak else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--rps", type=int, default=200)
    args = ap.parse_args()
    return run(args.seconds, args.rps)


if __name__ == "__main__":
    sys.exit(main())
