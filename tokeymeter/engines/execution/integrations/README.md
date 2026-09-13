# Tokeymeter integrations — drop-in wrappers

Adopt Tokeymeter without restructuring your code. Wrap the provider client you
already use, and every call it makes is optimized and recorded.

## OpenAI

```python
from openai import OpenAI
from tokeymeter.integrations.openai import wrap

client = wrap(OpenAI())
r = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "hello"}],
)
```

## Anthropic

```python
import anthropic
from tokeymeter.integrations.anthropic import wrap

client = wrap(anthropic.Anthropic())
r = client.messages.create(
    model="claude-haiku-4", max_tokens=256,
    messages=[{"role": "user", "content": "hello"}],
)
```

## What `wrap()` does

On `chat.completions.create` (OpenAI) / `messages.create` (Anthropic):

- **Exact cache** — a byte-identical request returns the stored response. No
  model call. Cost zero, latency a local lookup.
- **Semantic cache** (on by default; needs `tokeymeter[semantic]`) — a reworded
  but equivalent request returns the stored answer, gated by a confidence
  threshold with a safe fallback to a fresh call.
- **Single-flight** — identical concurrent requests collapse to one model call
  whose result is shared.
- **Content-blind audit** — a tamper-evident record of every optimization, keyed
  by a one-way hash of the request. No prompt text is stored.
- **Real-usage reconciliation** — on calls that actually run, the provider's own
  reported token counts (`response.usage`) are recorded so the savings figure
  reconciles against your bill.

## Options

```python
wrap(client,
     semantic=True,            # enable the near-duplicate cache (default True)
     semantic_threshold=0.92,  # similarity cutoff for a semantic hit
     shadow=False,             # measure-only: compute what WOULD be saved
     tag="support")            # label for per-team / per-feature breakdowns
```

`shadow=True` is the risk-free way to quantify value on live traffic: the engine
computes what it *would* have saved without serving any cached response, so you
can see the upside before trusting the cache with real answers.

## What is intentionally not cached

Streaming responses (`stream=True`) and tool-calling responses pass straight
through to the provider, uncached — a streamed or tool-invoking result is not a
deterministic, reusable value. The wrapper never changes their behavior.

## Fail-open guarantee

If anything inside the engine raises — a backend is unreachable, a serializer
errors — the wrapper calls your original client method and returns its result.
The wrapper is incapable, by design, of being the reason a call fails.

## The savings report

```bash
pip install "tokeymeter[providers,audit]"
tokeymeter report --demo                # no key, no spend — preview the format
tokeymeter report --provider openai     # real numbers from OPENAI_API_KEY
```

Writes a signed, self-contained `tokeymeter-savings.html` (and `.json`). The
figure is the per-call cost of the calls that ran, applied to the calls that
didn't, with the provider's reported token usage as the basis. Reconcile it
against your provider invoice for the authoritative bill.

You can also build a report programmatically:

```python
from tokeymeter import reporting

rec = reporting.validate("openai")            # or pass client=<your client>
path = reporting.render_html(rec, "report.html")
```
