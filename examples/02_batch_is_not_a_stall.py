"""Why response novelty ALONE would be a false-alarm machine.

A classifier answering "APPROVED" five hundred times has the same novelty score
as a completely stuck agent. The difference is the input: a stuck conversational
agent re-sends its history and grows, while independent batch work stays flat.

Both signals are required. This file shows what happens with each shape.
"""
import tokeymeter
from tokeymeter.storage import MemoryStore
from tokeymeter.engines.economics.usage import set_reported_usage
from tokeymeter.engines.governance.agents import agent_report, render_agents

tokeymeter.set_default_store(MemoryStore())
tokeymeter.set_in_memory_savings(True)
tokeymeter.register_pricing("gpt-4o", input_per_1m=2.5, output_per_1m=10.0)


@tokeymeter.cache(model="gpt-4o")
def call(prompt, tokens, answer):
    set_reported_usage(tokens, 200)
    return answer


# A classifier: same answer, INDEPENDENT documents, so input stays flat.
for i in range(12):
    with tokeymeter.task(f"doc-{i}", agent="invoice-classifier",
                         stall_window=8, enforce=True):
        for j in range(10):
            call(f"invoice-{i}-{j}", 1400, "APPROVED")

# A stuck agent: same answer, GROWING conversation.
history = []
stalled = 0
for i in range(4):
    try:
        with tokeymeter.task(f"ticket-{i}", agent="support",
                             stall_window=8, enforce=True):
            for j in range(14):
                history.append(f"retry {j}")
                call(tuple(history) + (i,), 1100 + j * 420, "tool error")
    except tokeymeter.TaskStalled:
        stalled += 1

print(render_agents(agent_report()))
print(f"\nThe classifier scored low on novelty and was correctly left alone.")
print(f"The support agent was stopped {stalled} times.")
