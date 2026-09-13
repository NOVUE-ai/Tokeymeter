"""The core case: an agent that has stopped getting anywhere.

A support agent hits a document it cannot parse. It retries. Every retry
re-sends the whole conversation, so each one costs more than the last while
learning nothing new.

Run this file. No API key needed.
"""
import tokeymeter
from tokeymeter.storage import MemoryStore
from tokeymeter.engines.economics.usage import set_reported_usage

tokeymeter.set_default_store(MemoryStore())
tokeymeter.set_in_memory_savings(True)
tokeymeter.register_pricing("gpt-4o", input_per_1m=2.5, output_per_1m=10.0)


@tokeymeter.cache(model="gpt-4o")
def agent_turn(messages, tokens):
    """Stand-in for your model call. The tool is broken, so the answer never
    changes — which is exactly what a stuck agent looks like."""
    set_reported_usage(tokens, 300)
    return "tool error: cannot parse attachment"


def run(ceiling: bool):
    history = []
    limits = dict(stall_window=8, enforce=True) if ceiling else {}
    with tokeymeter.task("ticket-88214", agent="support", **limits) as t:
        try:
            for i in range(40):
                history.append(f"assistant: retrying extraction, attempt {i}")
                agent_turn(tuple(history), 1100 + i * 420)
        except tokeymeter.TaskStalled as exc:
            return t.snapshot(), str(exc)
    return t.snapshot(), None


print("WITHOUT A CEILING")
before, _ = run(ceiling=False)
print(f"  {before['calls']} calls, ${before['spend_usd']:.4f}, 0 tickets resolved")
print("  Nothing anywhere says the agent achieved nothing.\n")

# A cold store for the second run. Sharing one would serve the whole second run
# from the first run's cache — every call free, nothing to halt, and a demo that
# quietly proves nothing.
tokeymeter.set_default_store(MemoryStore())
tokeymeter.reset_savings()

print("WITH ONE LINE")
after, message = run(ceiling=True)
print(f"  {after['calls']} calls, ${after['spend_usd']:.4f}")
print(f"  {message}")
