"""What number should stall_window be?

Guess low and healthy work dies; guess high and the runaway slips through. So
most people set nothing, and a ceiling nobody sets protects nobody.

`--suggest` answers it from your own history: every candidate is replayed and
reported with the two counts that decide it — stalled tasks caught, and HEALTHY
tasks it would also have stopped.
"""
import tokeymeter
from tokeymeter.storage import MemoryStore
from tokeymeter.engines.economics.usage import set_reported_usage
from tokeymeter.engines.governance.suggest import (
    suggest_thresholds, render_suggestions)

tokeymeter.set_default_store(MemoryStore())
tokeymeter.set_in_memory_savings(True)
tokeymeter.register_pricing("gpt-4o", input_per_1m=2.5, output_per_1m=10.0)


@tokeymeter.cache(model="gpt-4o")
def call(prompt, tokens, answer):
    set_reported_usage(tokens, 300)
    return answer


# Record first, with no ceilings. This is the right way to start: it cannot
# break anything, and it collects what the recommendation is fitted to.
for i in range(20):                                   # healthy
    history = []
    with tokeymeter.task(f"ok-{i}", agent="support"):
        for j in range(9):
            history.append(f"step {j}")
            call(tuple(history) + (i,), 900 + j * 330, f"found {i}-{j}")

for i in range(4):                                    # stuck
    history = []
    with tokeymeter.task(f"bad-{i}", agent="support"):
        for j in range(14):
            history.append(f"retry {j}")
            call(tuple(history) + (i, "x"), 1100 + j * 420, "tool error")

for i in range(10):                                   # batch — different shape
    with tokeymeter.task(f"doc-{i}", agent="extract"):
        for j in range(8):
            call(f"doc-{i}-{j}", 1400, "APPROVED")

print(render_suggestions(suggest_thresholds()))
print("\nNote what it refuses to answer, and why. A recommendation the data")
print("cannot support is worse than none.")
