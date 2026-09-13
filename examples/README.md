# Examples

Every file here runs on its own with **no API key and no network** — they use a
scripted model so you can see the behaviour before wiring anything real.

```bash
python examples/01_stuck_agent.py
```

| | |
|---|---|
| `01_stuck_agent.py` | the core case: an agent that stops getting anywhere, and what stopping it looks like |
| `02_batch_is_not_a_stall.py` | why novelty alone would be wrong, shown with a classifier |
| `03_choosing_thresholds.py` | `--suggest`, and how a threshold is backtested against real history |
| `04_alerting.py` | `on_halt`, including the case where your own code swallows the exception |
| `05_policy_file.py` | moving the numbers out of code into a file your platform team owns |
| `06_compliance.py` | model allowlists, data residency, and the report an auditor asks for |

To use these against a real provider, replace the scripted function with a
wrapped client — that is the only change:

```python
from tokeymeter.engines.execution.integrations import openai as tokeymeter_openai
client = tokeymeter_openai.wrap(OpenAI())
```
