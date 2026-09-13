"""`tokeymeter firstrun` — see the thing this exists for, in about a minute.

WHY THIS IS NOT THE OTHER DEMO
------------------------------
`tokeymeter demo` shows the optimization layer: caching, collapse, savings. That
was the old front door, and it opens on the wrong room. The first number a new
user saw was a cache hit rate and a fraction of a cent saved — true, unimpressive,
and indistinguishable from every other tool in the category.

This shows what nothing else can: an agent that is stuck, and what happens when
something is finally allowed to stop it.

Everything here runs offline against a scripted fake model. No API key, no
account, no network. The numbers printed are computed live from the same ledger
and the same enforcement path a real workload uses — nothing is hard-coded.
"""
from __future__ import annotations

import sys
import os
from typing import Optional


def _rule() -> str:
    return "-" * 66


def main(argv: Optional[list] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("-h", "--help"):
        print("tokeymeter firstrun - watch a stuck agent get stopped\n\n"
              "  tokeymeter firstrun\n\n"
              "Runs entirely offline against a scripted model. No key, no account.")
        return 0

    import tokeymeter
    from tokeymeter.storage import MemoryStore
    from tokeymeter.engines.economics.usage import set_reported_usage
    from tokeymeter.engines.governance import rules as _rules
    from tokeymeter.engines.governance.agents import agent_report, render_agents

    # ISOLATE COMPLETELY BEFORE TOUCHING ANYTHING.
    #
    # This is a demo, and a demo that damages real data is worse than no demo.
    # It resets the savings tracker to show a clean before/after, and without
    # isolation that reset lands on the ledger the user is actually
    # accumulating — verified: running this wiped a real savings.jsonl.
    #
    # So: a scratch ledger path for the duration, restored on the way out even
    # if this raises.
    import tempfile
    scratch = tempfile.mkdtemp(prefix="tokeymeter-firstrun-")
    previous_path = None
    try:
        from tokeymeter import paths as _paths
        previous_path = _paths._savings_override
    except Exception:
        pass
    try:
        tokeymeter.set_savings_path(os.path.join(scratch, "firstrun.jsonl"))
    except Exception:
        pass
    tokeymeter.set_default_store(MemoryStore())
    tokeymeter.set_in_memory_savings(True)
    tokeymeter.reset_savings()
    tokeymeter.clear_registered_pricing()
    tokeymeter.register_pricing("gpt-4o", input_per_1m=2.5, output_per_1m=10.0)
    _rules.clear_rules()

    upstream = {"n": 0}

    # A support agent working a ticket. Its tool cannot parse one document, so
    # it retries — and because the conversation history grows every turn, each
    # retry costs MORE than the last while learning nothing new.
    @tokeymeter.cache(model="gpt-4o")
    def agent_turn(messages, tokens):
        upstream["n"] += 1
        set_reported_usage(tokens, 300)
        return "tool error: cannot parse attachment"

    def run_ticket(*, ceiling: bool):
        # A COLD cache for each run. Both runs replay the same ticket, so
        # sharing a cache would serve the second run entirely from the first —
        # every call a free hit, nothing to halt, and a demo that quietly
        # proves nothing. Two independent runs is what is being shown.
        tokeymeter.set_default_store(MemoryStore())
        tokeymeter.reset_savings()
        upstream["n"] = 0
        history = []
        stopped = None
        kwargs = dict(stall_window=8, enforce=True) if ceiling else {}
        with tokeymeter.task("ticket-88214", agent="support", **kwargs) as t:
            try:
                for i in range(40):
                    history.append(f"assistant: retrying extraction, attempt {i}")
                    agent_turn(tuple(history), 1100 + i * 420)
            except tokeymeter.TaskStalled as exc:
                stopped = str(exc)
        return t.snapshot(), stopped

    print()
    print("  TOKEYMETER - first run".ljust(66))
    print(_rule())
    print("  A support agent is working ticket-88214. Its document parser is")
    print("  failing, so the agent retries. Nothing is wrong with your code.")
    print()

    before, _ = run_ticket(ceiling=False)
    spend_before = before["spend_usd"]
    calls_before = before["calls"]
    print("  WITHOUT A CEILING")
    print(f"    calls made          {calls_before}")
    print(f"    spent               ${spend_before:.4f}")
    print(f"    tickets resolved    0")
    print()
    print("    Your dashboard shows a completed task. Your bill shows a normal")
    print("    number. Nothing anywhere says the agent achieved nothing.")
    print()

    after, message = run_ticket(ceiling=True)
    spend_after = after["spend_usd"]

    print("  WITH ONE LINE")
    print()
    print("    with tokeymeter.task('ticket-88214', agent='support',")
    print("                         stall_window=8, enforce=True):")
    print("        agent.run(ticket)")
    print()
    print(f"    calls made          {after['calls']}   (was {calls_before})")
    print(f"    spent               ${spend_after:.4f}   (was ${spend_before:.4f})")
    if message:
        print()
        print("    It stopped, and said why:")
        for line in _wrap(message, 58):
            print(f"      {line}")
    print()
    print(_rule())
    print("  HOW IT KNEW")
    print()
    print("    Not from the cost - a stuck agent repeats itself, and repeats are")
    print("    served from cache, so looping is nearly free.")
    print("    Not from the prompt - the conversation grows every turn, so every")
    print("    prompt looks new.")
    print()
    print("    From the RESPONSES. A working agent returns new information; a")
    print("    stuck one returns the same answer again. Responses are hashed,")
    print("    never read.")
    print()

    rep = agent_report()
    print(_rule())
    print("  AND ACROSS EVERY AGENT YOU WRAP")
    print()
    for line in render_agents(rep).splitlines()[:8]:
        print(f"    {line}")
    print()
    print(_rule())
    print("  NEXT")
    print()
    # BOTH lines, in order. Showing only the task block was a dead end: a
    # reader followed it literally, got zero records, and `tokeymeter agents`
    # then told them to wrap an entry point — the very thing they had just
    # done. The client wrap is what puts calls on the ledger; the task block is
    # what groups them.
    print("    1  wrap your client once, where you create it:")
    print()
    print("       from tokeymeter.engines.execution.integrations import \\")
    print("           openai as tokeymeter_openai")
    print("       client = tokeymeter_openai.wrap(OpenAI())")
    print()
    print("    2  put a task boundary around one unit of work:")
    print()
    print("       with tokeymeter.task(f'ticket-{id}', agent='support'):")
    print("           agent.run(ticket)")
    print()
    print("       (start without enforce=True - it only records, so it cannot")
    print("        break anything, and it collects what step 4 needs)")
    print()
    print("    3  tokeymeter agents          - cost per task, and what stalled")
    print("    4  tokeymeter agents --suggest - what to set, backtested on")
    print("                                     your own traffic")
    print()
    print("    Nothing left this process. No prompt or response was stored.")
    print("    Your own ledger was not touched - this ran in a scratch file.")
    print()
    _rules.clear_rules()
    _restore(tokeymeter, previous_path, scratch)
    return 0


def _restore(tokeymeter, previous_path, scratch) -> None:
    """Put the user's ledger path back and remove the scratch file."""
    try:
        if previous_path is not None:
            tokeymeter.set_savings_path(previous_path)
    except Exception:
        pass
    try:
        import shutil
        shutil.rmtree(scratch, ignore_errors=True)
    except Exception:
        pass


def _wrap(text: str, width: int):
    words, line, out = text.split(), "", []
    for w in words:
        if len(line) + len(w) + 1 > width:
            out.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        out.append(line)
    return out
