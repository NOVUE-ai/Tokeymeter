"""Routing halts to wherever your team already looks.

The important detail: this fires even when YOUR code swallows the exception.
`TaskLimitExceeded` subclasses `RuntimeError`, so ordinary retry code catches it
— measured, 22 halts swallowed on one stuck agent while the task was correctly
bounded at 8 calls. A hook riding on the exception would be silent in exactly
the case you need to hear about.

Replace `notify` with an HTTP POST to a Slack webhook, a Jira client, or
anything else you already have configured. There are no bundled connectors on
purpose — each would be a dependency, an auth scheme and a token store in a
package that currently has none.
"""
import tokeymeter
from tokeymeter.storage import MemoryStore
from tokeymeter.engines.economics.usage import set_reported_usage

tokeymeter.set_default_store(MemoryStore())
tokeymeter.set_in_memory_savings(True)
tokeymeter.register_pricing("gpt-4o", input_per_1m=2.5, output_per_1m=10.0)


@tokeymeter.on_halt
def notify(halt):
    """Registered once, process-wide. Fires once per halted task.

    In production this would be:
        urlopen(Request(SLACK_WEBHOOK,
                        data=json.dumps({"text": halt.summary()}).encode(),
                        headers={"Content-Type": "application/json"}))
    """
    print(f"  ALERT  {halt.summary()}")


@tokeymeter.cache(model="gpt-4o")
def agent_turn(messages, tokens):
    set_reported_usage(tokens, 300)
    return "tool error: cannot parse"


history = []
swallowed = 0
with tokeymeter.task("ticket-88214", agent="support",
                     stall_window=8, enforce=True) as t:
    for i in range(30):
        history.append(f"retry {i}")
        try:
            agent_turn(tuple(history), 1000 + i * 400)
        except Exception:              # what real agent retry code does
            swallowed += 1

print(f"\n  the agent swallowed {swallowed} halts and never noticed")
print(f"  the task was still bounded at {t.snapshot()['calls']} calls")
print("  and the alert above fired anyway — exactly once")
