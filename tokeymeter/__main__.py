"""
Tokeymeter command-line entry point.

    tokeymeter firstrun                 # 60 seconds: watch a stuck agent stop
    tokeymeter watch                    # see tasks as they run, live
    tokeymeter agents                   # how your agents behave: cost per task,
                                        #   progress, and which ones stalled
    tokeymeter plan --policy FILE       # what an execution policy WOULD do,
                                        #   replayed against your own history
    tokeymeter doctor                   # verdict + overhead p50/p95/p99 + ledger health
    tokeymeter report --demo            # savings report (no keys)
    tokeymeter report --provider openai # against your real OPENAI_API_KEY
    tokeymeter demo                     # self-hosted walkthrough (offline, no keys)
    tokeymeter pricing                  # list-table age + registered overrides
    tokeymeter version

`python -m tokeymeter ...` and the installed `tokeymeter` script both land here.
"""
from __future__ import annotations

import sys


def _make_console_resilient() -> None:
    """Ensure CLI output can never crash on a legacy console code page.

    Windows consoles still default to cp1252 in many environments, and a single
    non-ASCII character in an otherwise fine report (an arrow, a middle dot)
    raises UnicodeEncodeError and takes the whole command down. `tokeymeter
    demo` did exactly that on '\\u2192'.

    We keep the stream's own encoding — rewriting it to UTF-8 would emit bytes a
    cp1252 console renders as mojibake — and only switch the error handler to
    'replace'. On a UTF-8 terminal output is unchanged; on a legacy one an
    unrepresentable glyph degrades to '?' instead of aborting the command.
    A diagnostic tool that dies while printing diagnostics is worse than one
    that prints an imperfect character."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:      # not a TextIOWrapper (piped/captured)
            continue
        try:
            reconfigure(errors="replace")
        except Exception:
            # Console setup must never be the reason a command fails.
            pass


def _print_doctor() -> None:
    """Render tokeymeter.doctor() — one-line verdict first, then cache-hit
    overhead percentiles, savings mode, ledger health, degraded counts."""
    import json
    import tokeymeter
    d = tokeymeter.doctor()
    oh = d["cache_hit_overhead"]
    h = d["ledger_health"]

    def ms(v):
        return "n/a" if v is None else f"{v}ms"

    degraded = bool(h.get("degraded")) or (d.get("degraded_events", 0) or 0) > 0
    if degraded:
        why = []
        if h.get("degraded"):
            why.append("ledger degraded")
        if (d.get("degraded_events", 0) or 0) > 0:
            why.append(f"{d['degraded_events']} degraded event(s)")
        verdict = "DEGRADED — " + ", ".join(why)
    else:
        verdict = "HEALTHY"

    print(f"tokeymeter doctor  |  {verdict}")
    print("  cache-hit overhead:  "
          f"p50={ms(oh['p50_ms'])}  p95={ms(oh['p95_ms'])}  "
          f"p99={ms(oh['p99_ms'])}  (n={oh['samples']})")
    print(f"  savings mode:        {d['savings_mode']}")
    print(f"  ledger:              writable={h['writable']}  degraded={h['degraded']}  "
          f"records={h['records_written']}  write_failures={h['write_failures']}")
    print(f"  degraded events:     {d['degraded_events']}  by_source={d['degraded_by_source']}")
    # machine-readable line for scripting (kept stable)
    print(json.dumps(d))


def main(argv=None) -> None:
    _make_console_resilient()
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__.strip())
        return
    cmd, rest = argv[0], argv[1:]
    if cmd == "report":
        from tokeymeter import reporting
        sys.argv = ["tokeymeter report", *rest]
        reporting.main()
    elif cmd == "doctor":
        _print_doctor()
    elif cmd == "pricing":
        from tokeymeter import pricing as _pr
        age = _pr.pricing_age_days()
        stale = age > 45
        print(f"tokeymeter pricing  |  list table as of {_pr.PRICING_AS_OF} "
              f"({age}d old){'  |  STALE - refresh recommended' if stale else ''}")
        reg = _pr.registered_pricing()
        if reg:
            print(f"  registered overrides ({len(reg)}):")
            for m, e in sorted(reg.items()):
                print(f"    {m:<28} in=${e['input']:.4f}/1M  out=${e['output']:.4f}/1M")
        else:
            print("  registered overrides: none")
        print("  note: registered rates always take precedence over the list table.")
    elif cmd == "firstrun":
        from tokeymeter import firstrun as _firstrun
        sys.exit(_firstrun.main(rest))
    elif cmd == "watch":
        from tokeymeter import watch as _watch
        sys.exit(_watch.main(rest))
    elif cmd == "agents":
        from tokeymeter.engines.governance import agents as _agents
        sys.exit(_agents.main(rest))
    elif cmd == "plan":
        from tokeymeter.engines.governance import plan as _plan
        sys.exit(_plan.main(rest))
    elif cmd == "demo":
        from tokeymeter import demo
        sys.exit(demo.run())
    elif cmd in ("version", "--version", "-V"):
        import tokeymeter
        print("tokeymeter", getattr(tokeymeter, "__version__", "unknown"))
    else:
        print(f"unknown command: {cmd}\n")
        print(__doc__.strip())
        sys.exit(2)


if __name__ == "__main__":
    main()
