"""
ACCEPTANCE.py — run this on YOUR Windows machine, in VS Code, before publishing.

WHAT THIS IS FOR
================
Every workload this product has been tested against was written by the same
person who wrote the product. That is the one gap no amount of test coverage
closes. This script walks the actual journey a new user takes, on a real
Windows console, and reports two different things:

  PASS/FAIL   does it work
  FRICTION    what a first-time user would find confusing, slow, or missing

The second column is the point. A green suite tells you the code is correct.
It cannot tell you the product is usable.

HOW TO RUN
==========
In VS Code, open a terminal in an empty folder and:

    python -m venv .venv
    .venv\\Scripts\\activate
    pip install <path-to>\\novue-tokeymeter-<version>.tar.gz
    python ACCEPTANCE.py

Run it a SECOND time in a plain `cmd.exe` window (not the VS Code terminal),
because cmd.exe defaults to a legacy code page and that is where console
encoding problems appear.

Nothing here needs an API key, a network connection, or an account.
"""
from __future__ import annotations

import io
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time

RESULTS = []
FRICTION = []


def check(name, fn, *, note=None):
    """Run one step and record the outcome without ever stopping the walk —
    a first-time user does not get to skip the rest of the product because
    step three raised."""
    t0 = time.perf_counter()
    try:
        detail = fn()
        ok = True
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        ok = False
    ms = (time.perf_counter() - t0) * 1000
    RESULTS.append((ok, name, detail, ms))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<44} {ms:7.0f} ms")
    if detail and (not ok or note == "show"):
        for line in str(detail).splitlines()[:6]:
            print(f"         {line}")
    return ok


def friction(what):
    FRICTION.append(what)


def run_cli(*args, timeout=180):
    """Invoke the CLI the way a user does, capturing BYTES.

    Decoding as UTF-8 would fail on correctly-encoded legacy-code-page output
    and report a crash that never happened — the claim being tested is "the
    command completes", not "the output is UTF-8".
    """
    p = subprocess.run([sys.executable, "-m", "tokeymeter", *args],
                       capture_output=True, timeout=timeout)
    out = (p.stdout or b"").decode("utf-8", "replace")
    err = (p.stderr or b"").decode("utf-8", "replace")
    return p.returncode, out + err


def main() -> int:
    print()
    print("=" * 72)
    print("  TOKEYMETER — user-perspective acceptance")
    print("=" * 72)
    print(f"  python      {sys.version.split()[0]}")
    print(f"  platform    {platform.platform()}")
    print(f"  stdout enc  {getattr(sys.stdout, 'encoding', '?')}")
    print(f"  cwd         {os.getcwd()}")
    print()

    home = tempfile.mkdtemp(prefix="tokeymeter-acceptance-")
    os.environ["TOKEYMETER_HOME"] = home
    print(f"  scratch     {home}")
    print()

    # ── 1. it installed and is honest about what it is ──────────────────
    print("-- 1. install and identity " + "-" * 45)
    import tokeymeter

    check("import tokeymeter", lambda: f"v{tokeymeter.__version__}", note="show")
    check("the package verifies its own integrity",
          lambda: __import__(
              "tokeymeter.engines.trust.integrity", fromlist=["x"]
          ).verify_self(require_signature=False).status,
          note="show")
    check("a newcomer can see what the product IS",
          lambda: f"{len(tokeymeter.PRIMARY_API)} primary of "
                  f"{len(tokeymeter.__all__)} public names", note="show")

    # ── 2. the first run ────────────────────────────────────────────────
    print()
    print("-- 2. the first 60 seconds " + "-" * 45)

    def first_run():
        code, out = run_cli("firstrun")
        if code != 0:
            raise RuntimeError(f"exit {code}\n{out[:400]}")
        if "was" not in out or "stopped" not in out.lower():
            raise RuntimeError("the demo did not show a before/after")
        return "showed a stuck agent being stopped"

    check("tokeymeter firstrun", first_run, note="show")

    def help_discovers_the_product():
        code, out = run_cli()
        missing = [c for c in ("firstrun", "agents", "plan", "doctor")
                   if c not in out]
        if missing:
            raise RuntimeError(f"help does not mention: {', '.join(missing)}")
        return "help lists the commands that matter"

    check("the CLI tells you what it can do", help_discovers_the_product)

    # ── 3. the developer's own path ─────────────────────────────────────
    print()
    print("-- 3. wrapping one agent " + "-" * 47)
    from tokeymeter.storage import MemoryStore
    from tokeymeter.engines.economics.usage import set_reported_usage

    tokeymeter.set_default_store(MemoryStore())
    tokeymeter.set_savings_path(os.path.join(home, "savings.jsonl"))
    tokeymeter.register_pricing("gpt-4o", input_per_1m=2.5, output_per_1m=10.0)

    @tokeymeter.cache(model="gpt-4o", tag="support-api")
    def agent_turn(messages, tokens, stuck, i, j):
        set_reported_usage(tokens, 300)
        return "tool error: cannot parse" if stuck else f"found {i}-{j}"

    def wrap_and_halt():
        stopped = 0
        for i in range(12):
            history = []
            try:
                with tokeymeter.task(f"ticket-{i}", agent="support",
                                     stall_window=8, enforce=True):
                    for j in range(14):
                        history.append(f"turn {j}")
                        agent_turn(tuple(history) + (i,), 1100 + j * 420,
                                   i % 4 == 0, i, j)
            except tokeymeter.TaskStalled:
                stopped += 1
        if stopped == 0:
            raise RuntimeError("no stuck task was stopped")
        return f"{stopped} of 12 tickets stopped as stuck"

    check("a stuck agent is halted", wrap_and_halt, note="show")

    def report_card():
        from tokeymeter.engines.governance.agents import (agent_report,
                                                          render_agents)
        text = render_agents(agent_report())
        if "support" not in text:
            raise RuntimeError("the agent is missing from its own report")
        non_ascii = [c for c in text if ord(c) > 127]
        if non_ascii:
            raise RuntimeError(f"report contains non-ASCII: {non_ascii[:5]!r}")
        return text.splitlines()[2].strip()

    check("tokeymeter agents shows the estate", report_card, note="show")

    # ── 4. the platform owner's path ────────────────────────────────────
    print()
    print("-- 4. one policy across every service " + "-" * 34)
    policy_path = os.path.join(home, "ai-execution.json")
    with open(policy_path, "w", encoding="utf-8") as f:
        json.dump({"version": 1, "rules": [
            {"name": "agent-guard",
             "then": {"envelope": 2.00, "reserve": 0.10,
                      "stall_window": 8, "enforce": True}},
            {"name": "phi-handling", "when": {"data_class": "PHI"},
             "then": {"only": ["gpt-4o-secure"], "never_cache": True}},
        ]}, f)

    def plan_runs():
        code, out = run_cli("plan", "--policy", policy_path)
        if code not in (0, 2):
            raise RuntimeError(f"exit {code}\n{out[:400]}")
        if "Coverage" not in out:
            raise RuntimeError("plan did not report coverage")
        return next((l.strip() for l in out.splitlines()
                     if "would halt" in l), "ran")

    check("tokeymeter plan previews a policy", plan_runs, note="show")

    def plan_gates_ci():
        code, _ = run_cli("plan", "--policy", policy_path,
                          "--protect", "support")
        if code not in (0, 2):
            raise RuntimeError(f"unexpected exit {code}")
        return f"exit {code} (2 means it would block a protected agent)"

    check("plan can fail a CI step", plan_gates_ci, note="show")

    def bad_policy_is_loud():
        bad = os.path.join(home, "bad.json")
        with open(bad, "w", encoding="utf-8") as f:
            json.dump({"rules": [{"name": "oops",
                                  "then": {"enevelope": 1.0}}]}, f)
        code, out = run_cli("plan", "--policy", bad)
        if code == 0:
            raise RuntimeError("a typo'd policy was accepted silently")
        if "oops" not in out:
            raise RuntimeError("the error does not name the offending rule")
        return "a typo is rejected and the rule is named"

    check("a bad policy fails loudly", bad_policy_is_loud, note="show")

    # ── 5. the compliance officer's path ────────────────────────────────
    print()
    print("-- 5. what is allowed, and proving it " + "-" * 34)
    from tokeymeter.engines.governance import rules as R

    def phi_is_refused():
        R.set_rules(R.load_rules([
            {"name": "phi", "when": {"data_class": "PHI"},
             "then": {"only": ["gpt-4o-secure"], "never_cache": True}}]))
        tokeymeter.register_pricing("gpt-4o-secure", input_per_1m=2.5,
                                    output_per_1m=10.0)

        @tokeymeter.cache(model="gpt-4o")
        def unapproved(p):
            set_reported_usage(1000, 200)
            return "x"

        @tokeymeter.cache(model="gpt-4o-secure")
        def approved(p):
            set_reported_usage(1000, 200)
            return "x"

        refused = 0
        for i in range(5):
            try:
                with tokeymeter.data_class("PHI"):
                    unapproved(f"patient {i}")
            except tokeymeter.ModelNotPermitted:
                refused += 1
        for i in range(10):
            with tokeymeter.data_class("PHI"):
                approved(f"patient {i}")
        if refused != 5:
            raise RuntimeError(f"only {refused} of 5 were refused")
        return "5 of 5 attempts to send PHI to an unapproved model refused"

    check("PHI cannot reach an unapproved model", phi_is_refused, note="show")

    def auditor_answer():
        rep = tokeymeter.policy_report()
        models = rep["by_data_class"].get("PHI", {}).get("models")
        if models != ["gpt-4o-secure"]:
            raise RuntimeError(f"PHI reached: {models}")
        return f"every model PHI reached: {models}"

    check("the report answers 'prove it'", auditor_answer, note="show")

    def nothing_is_stored():
        recs = [json.loads(l) for l in
                open(os.path.join(home, "savings.jsonl"), encoding="utf-8")
                if l.strip()]
        blob = json.dumps(recs)
        leaked = [s for s in ("patient", "tool error", "found 0-0", "turn 1")
                  if s in blob]
        if leaked:
            raise RuntimeError(f"content found in the ledger: {leaked}")
        return f"{len(recs)} records on disk, none containing prompt or response"

    R.clear_rules()
    check("no prompt or response reached the ledger", nothing_is_stored,
          note="show")

    # ── 6. Windows-specific reality ─────────────────────────────────────
    print()
    print("-- 6. this machine, specifically " + "-" * 39)

    def console_survives_legacy_codepage():
        env = dict(os.environ, PYTHONIOENCODING="cp1252")
        bad = []
        for cmd in ("firstrun", "doctor", "pricing", "agents"):
            p = subprocess.run([sys.executable, "-m", "tokeymeter", cmd],
                               capture_output=True, env=env, timeout=300)
            if p.returncode != 0:
                bad.append(f"{cmd}(exit {p.returncode})")
        if bad:
            raise RuntimeError("crashed under cp1252: " + ", ".join(bad))
        return "every command survives a legacy console code page"

    check("CLI works on a legacy code page", console_survives_legacy_codepage,
          note="show")

    def home_is_respected():
        found = [f for f in os.listdir(home) if f.endswith(".jsonl")]
        if not found:
            raise RuntimeError("nothing was written to TOKEYMETER_HOME")
        stray = os.path.join(os.path.expanduser("~"), ".tokeymeter")
        note = ""
        if os.path.isdir(stray):
            note = "  (note: ~/.tokeymeter also exists from an earlier run)"
        return f"wrote {found} into TOKEYMETER_HOME{note}"

    check("TOKEYMETER_HOME is respected", home_is_respected, note="show")

    def ledger_write_speed():
        tokeymeter.set_default_store(MemoryStore())

        @tokeymeter.cache(model="gpt-4o")
        def s(p):
            set_reported_usage(400, 150)
            return "x"

        t0 = time.perf_counter()
        for i in range(2000):
            with tokeymeter.task(f"perf-{i}", agent="perf"):
                s(f"{i}")
        rate = 2000 / (time.perf_counter() - t0)
        if rate < 100:
            friction(f"ledger writes are slow here ({rate:,.0f}/s). On Windows "
                     f"this is usually antivirus scanning every file open — "
                     f"set_buffered_savings(True) cuts opens ~65x.")
        return f"{rate:,.0f} tasks/s writing to a real file"

    check("ledger throughput on this disk", ledger_write_speed, note="show")

    def overhead():
        tokeymeter.set_default_store(MemoryStore())
        tokeymeter.set_in_memory_savings(True)

        @tokeymeter.cache(model="gpt-4o")
        def s(p):
            set_reported_usage(400, 200)
            return "x" * 2000

        s("warm")
        lat = []
        for _ in range(2000):
            t0 = time.perf_counter_ns()
            with tokeymeter.task("bench", agent="b", stall_window=8):
                s("warm")
            lat.append((time.perf_counter_ns() - t0) / 1000)
        lat.sort()
        p50, p95 = lat[len(lat) // 2], lat[int(len(lat) * 0.95)]
        if p50 > 500:
            friction(f"per-call overhead is high here (p50 {p50:.0f}us).")
        return f"p50 {p50:.1f}us  p95 {p95:.1f}us  (fully governed call)"

    check("per-call overhead on this machine", overhead, note="show")

    # ── verdict ─────────────────────────────────────────────────────────
    passed = sum(1 for ok, *_ in RESULTS if ok)
    total = len(RESULTS)
    print()
    print("=" * 72)
    print(f"  {passed}/{total} checks passed")
    failed = [(n, d) for ok, n, d, _ in RESULTS if not ok]
    if failed:
        print()
        print("  FAILURES — these are what a user would hit:")
        for n, d in failed:
            print(f"    - {n}")
            for line in str(d).splitlines()[:3]:
                print(f"        {line}")
    if FRICTION:
        print()
        print("  FRICTION — works, but a user would notice:")
        for f in FRICTION:
            print(f"    - {f}")
    print()
    print("  WORTH ANSWERING YOURSELF, since no test can:")
    print("    1. After `tokeymeter firstrun`, did you know what to do next?")
    print("    2. Was anything on screen confusing or unexplained?")
    print("    3. Would you have wrapped one of your own agents after this?")
    print("    4. What did you expect to exist that did not?")
    print()
    print(f"  scratch dir left in place for inspection:\n    {home}")
    print("=" * 72)
    print()
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
