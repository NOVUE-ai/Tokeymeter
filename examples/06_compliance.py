"""Rules about what is PERMITTED, not just how much.

`data_class` and `region` are declared by your application and never inferred —
inferring means reading the prompt, which is the one thing this does not do. The
honest consequence: enforcement is on the DECLARED class, so a mis-tagged case
is faithfully governed by the wrong rule. Every record says "the declared rule
was applied to the declared class", never "we verified the content".
"""
import tokeymeter
from tokeymeter.storage import MemoryStore
from tokeymeter.engines.economics.usage import set_reported_usage
from tokeymeter.engines.execution.endpoint import endpoint
from tokeymeter.engines.governance import rules as R

tokeymeter.set_default_store(MemoryStore())
tokeymeter.set_in_memory_savings(True)
for m in ("gpt-4o", "gpt-4o-secure"):
    tokeymeter.register_pricing(m, input_per_1m=2.5, output_per_1m=10.0)

R.set_rules(R.load_rules({"rules": [
    {"name": "phi-handling", "when": {"data_class": "PHI"},
     "then": {"only": ["gpt-4o-secure"], "never_cache": True,
              "record_outcome": True}},
    {"name": "eu-residency", "when": {"region": "EU"},
     "then": {"only_endpoints": ["azure-westeurope"], "record_outcome": True}},
]}))


@tokeymeter.cache(model="gpt-4o")
def unapproved(prompt):
    set_reported_usage(1000, 200)
    return "answer"


@tokeymeter.cache(model="gpt-4o-secure")
def approved(prompt):
    set_reported_usage(1000, 200)
    return "answer"


refused = 0
for i in range(5):
    try:
        with tokeymeter.data_class("PHI"):
            unapproved(f"patient record {i}")
    except tokeymeter.ModelNotPermitted:
        refused += 1

for i in range(10):
    with tokeymeter.data_class("PHI"):
        approved(f"patient record {i}")

# Residency: the SAME model runs in several regions, so restricting the model
# name is not enough — only_endpoints restricts where a call is PROCESSED.
try:
    with tokeymeter.region("EU"), endpoint("azure-eastus"):
        approved("EU citizen record")
except tokeymeter.EndpointNotPermitted as exc:
    print(f"  residency refused: {exc}\n")

with tokeymeter.region("EU"), endpoint("azure-westeurope"):
    approved("EU citizen record")

report = tokeymeter.policy_report()
print(f"  {refused} attempts to send PHI to an unapproved model were refused\n")
print("  The auditor's question, answered from the ledger:")
print(f"    every model PHI reached : {report['by_data_class']['PHI']['models']}")
print(f"    where EU was processed  : {report['by_region']['EU']['endpoints']}")
R.clear_rules()
