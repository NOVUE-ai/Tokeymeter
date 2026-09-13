"""Seed a realistic agent estate so every command has something to show.

Run this ONCE before the walkthrough. It writes to ./tokeymeter-demo/ in the
current folder — nothing outside it is touched, and you can delete the folder
when you're done.

No API key. No network. A scripted model stands in for the real one, so the
numbers are real arithmetic over fake traffic.

The estate is deliberately mixed, because a screenshot of one agent proves
nothing. It contains:

  service-diagnostic  a conversational agent; a few tasks get stuck on a
                      document they cannot parse

Everything is seeded in RECORD-ONLY mode — no ceilings, nothing blocked. That
is what the README tells a new user to do first, and it is also why the report
card has something to show: a task that was HALTED at call 8 leaves behind eight
healthy-looking calls, so an estate that was governed from the start looks
suspiciously clean.
  warranty-triage     a batch classifier: the SAME answer every time with FLAT
                      input, which must NOT be flagged. This is the shot that
                      shows the detector has judgement.
  parts-lookup        single-shot calls, heavily cached
  contract-review     ran last week only, and got worse this week — so
                      `--compare` has something real to report
"""
import json
import os
import random
import sys
import time

os.environ["TOKEYMETER_HOME"] = os.path.abspath("tokeymeter-demo")
os.makedirs(os.environ["TOKEYMETER_HOME"], exist_ok=True)
LEDGER = os.path.join(os.environ["TOKEYMETER_HOME"], "savings.jsonl")

import tokeymeter                                             # noqa: E402
from tokeymeter.storage import MemoryStore                    # noqa: E402
from tokeymeter.engines.economics.usage import set_reported_usage  # noqa: E402

tokeymeter.set_default_store(MemoryStore())
tokeymeter.set_savings_path(LEDGER)
tokeymeter.register_pricing("gpt-4o", input_per_1m=2.5, output_per_1m=10.0)
tokeymeter.register_pricing("gpt-4o-mini", input_per_1m=0.15, output_per_1m=0.60)

rng = random.Random(11)


@tokeymeter.cache(model="gpt-4o", tag="cat-digital")
def turn(messages, tokens, answer):
    set_reported_usage(tokens, 300)
    return answer


@tokeymeter.cache(model="gpt-4o-mini", tag="cat-digital")
def lookup(part_number):
    set_reported_usage(220, 90)
    return f"part {part_number}: in stock"


def service_diagnostic(n=60):
    """Conversational. Every 9th task hits a bulletin it cannot parse."""
    for i in range(n):
        stuck = i % 9 == 0
        history = []
        try:
            with tokeymeter.task(f"diag-{4000 + i}", agent="service-diagnostic"):
                for j in range(rng.randint(9, 14)):
                    history.append(f"turn {j}")
                    turn(tuple(history) + (i,), 1100 + j * 420,
                         "tool error: cannot parse service bulletin" if stuck
                         else f"found fault code {i}-{j}")
        except tokeymeter.TaskStalled:
            pass


def warranty_triage(n=45):
    """Batch. Three labels, INDEPENDENT claims, so input stays flat.
    Low novelty and correctly NOT a stall."""
    for i in range(n):
        with tokeymeter.task(f"claim-{7100 + i}", agent="warranty-triage"):
            for j in range(6):
                turn(f"claim-{i}-{j}", 1400,
                     ["APPROVED", "DENIED", "REVIEW"][j % 3])


def parts_lookup(n=80):
    """Single-shot, and the same parts get asked for repeatedly."""
    parts = [f"P{rng.randint(100, 140)}" for _ in range(18)]
    for i in range(n):
        with tokeymeter.task(f"parts-{9000 + i}", agent="parts-lookup"):
            lookup(rng.choice(parts))


def contract_review(prefix, calls_per_task, n=14):
    """Runs in both windows, worse in the second one."""
    for i in range(n):
        history = []
        try:
            with tokeymeter.task(f"{prefix}-{i}", agent="contract-review"):
                for j in range(calls_per_task):
                    history.append(f"clause {j}")
                    turn(tuple(history) + (prefix, i), 1000 + j * 400,
                         "tool error: cannot parse" if (prefix == "cr-new"
                                                        and i % 5 == 0)
                         else f"clause {i}-{j} reviewed")
        except tokeymeter.TaskStalled:
            pass


print("seeding a realistic estate (no API key, no network)...")
contract_review("cr-old", 6)          # last week: healthy, 6 calls a task
service_diagnostic()
warranty_triage()
parts_lookup()
contract_review("cr-new", 11)         # this week: 11 calls a task, some stuck

# Backdate the "last week" tasks so `--since 7d --compare` has two real windows.
#
# SHIFT WHOLE TASKS, never individual records. The progress signal reads the
# first half of a task against the second, so assigning each record its own
# random timestamp would scramble the order inside a task and describe a
# sequence that never happened. (Found the hard way: it made 5 of 7 genuinely
# stuck tasks look healthy.) Real ledgers are append-ordered; this preserves
# that while moving tasks between windows.
now = time.time()
records = []
with open(LEDGER, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            try:
                records.append(json.loads(line))
            except ValueError:
                pass

task_offset = {}
seq = 0
for r in records:
    tid = str(r.get("task_id", ""))
    if tid not in task_offset:
        # one base time per task; older window for last week's contract review
        task_offset[tid] = now - (10 * 86400 if tid.startswith("cr-old")
                                  else rng.uniform(600, 3 * 86400))
    r["timestamp"] = task_offset[tid] + seq * 0.001   # keep append order intact
    seq += 1

with open(LEDGER, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r) + "\n")

print(f"  {len(records)} records written to {LEDGER}")
print("  4 agents, recorded with NO ceilings — exactly how you should start")
print()
print("Now point this shell at that ledger, then run the walkthrough:")
print()
if os.name == "nt":
    # BOTH forms, because `set VAR=...` is cmd.exe syntax and does nothing in
    # PowerShell — which is what VS Code opens by default. A reader who pastes
    # the wrong one gets an empty report and no clue why.
    print("  PowerShell (the VS Code default):")
    print(f'    $env:TOKEYMETER_HOME = "{os.environ["TOKEYMETER_HOME"]}"')
    print()
    print("  cmd.exe:")
    print(f'    set TOKEYMETER_HOME={os.environ["TOKEYMETER_HOME"]}')
else:
    print(f'    export TOKEYMETER_HOME="{os.environ["TOKEYMETER_HOME"]}"')
print()
print("  then:  tokeymeter agents")
