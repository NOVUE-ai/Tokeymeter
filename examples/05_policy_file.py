"""Moving the numbers out of code and into a file your platform team owns.

Hot-reloaded, no deploy. Between the file and your code the MOST RESTRICTIVE
wins, so neither side can loosen what the other set — which is what makes it
safe to hand the file to someone who does not read your code.
"""
import json
import os
import tempfile

import tokeymeter
from tokeymeter.storage import MemoryStore
from tokeymeter.engines.economics.usage import set_reported_usage
from tokeymeter.engines.governance import plan as P, rules as R

tokeymeter.set_default_store(MemoryStore())
tokeymeter.set_in_memory_savings(True)
tokeymeter.register_pricing("gpt-4o", input_per_1m=2.5, output_per_1m=10.0)

policy = {"version": 1, "rules": [
    {"name": "agent-guard",
     "then": {"envelope": 2.00, "reserve": 0.10, "stall_window": 8,
              "enforce": True}},
    {"name": "non-production-is-cheap",
     "when": {"env": ["dev", "ci"]}, "then": {"envelope": 0.25}},
    {"name": "checkout-is-protected",
     "when": {"agent": "checkout"}, "then": {"enforce": False}},
]}
path = os.path.join(tempfile.mkdtemp(), "ai-execution.json")
with open(path, "w", encoding="utf-8") as f:
    json.dump(policy, f)


@tokeymeter.cache(model="gpt-4o")
def call(prompt, tokens, answer):
    set_reported_usage(tokens, 300)
    return answer


# Build some ungoverned history first — that is what a plan is replayed against.
for i in range(10):
    history = []
    with tokeymeter.task(f"t-{i}", agent="support"):
        for j in range(12):
            history.append(f"r{j}")
            call(tuple(history) + (i,), 1000 + j * 400,
                 "tool error" if i % 3 == 0 else f"new {i}-{j}")

print("What this policy WOULD do, replayed against the history above:\n")
print(P.render_plan(P.plan_report(R.load_rules_file(path))))
print("\nOnly after reading that would you apply it:")
print("    tokeymeter.set_rules(tokeymeter.load_rules_file('ai-execution.json'))")
print("\nIn CI, `tokeymeter plan --policy FILE --protect checkout` exits 2 if a")
print("protected agent would be halted. A gate that cannot fail is decoration.")
