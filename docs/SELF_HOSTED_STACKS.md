# Tokeymeter on self-hosted stacks (vLLM · Ollama · TGI · llama.cpp · LM Studio)

If you serve open-source models on your own GPUs, Tokeymeter already works
with your stack — and it is the only class of tool that *can*: it runs
in-process, content-blind, with no proxy and no egress, which is compatible
with the exact constraints that made you self-host. This page is the whole
setup: wrap your call, register your true economics, read savings in the
unit that actually matters to you — **GPU capacity**.

Everything below uses only public Tokeymeter APIs (`pip install tokeymeter`,
v0.13+). No account, no telemetry, no data leaves your perimeter.

---

## 1. Wrap your call (any stack)

Tokeymeter wraps a *callable* — it does not care what is inside. All of the
stacks below expose an OpenAI-compatible HTTP API, so one pattern covers
them all; only `base_url` and the model name change.

```python
import tokeymeter
from openai import OpenAI  # pip install openai — used purely as an HTTP client

client = OpenAI(base_url="http://localhost:8000/v1",  # your serving endpoint
                api_key="not-needed-locally")

@tokeymeter.cache(model="kimi-k2-instruct")           # your served model name
def ask(prompt: str) -> str:
    resp = client.chat.completions.create(
        model="kimi-k2-instruct",
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content
```

Per-stack `base_url` defaults:

| Stack | Serve command (typical) | base_url |
|---|---|---|
| vLLM | `vllm serve <model>` | `http://localhost:8000/v1` |
| Ollama | `ollama serve` (runs by default) | `http://localhost:11434/v1` |
| TGI | `text-generation-launcher --model-id <model>` | `http://localhost:8080/v1` |
| llama.cpp | `llama-server -m model.gguf` | `http://localhost:8080/v1` |
| LM Studio | Developer → Local Server | `http://localhost:1234/v1` |

That's it for metering: exact + semantic caching, single-flight stampede
protection, the secret firewall, and the local savings ledger all apply to
your self-hosted calls exactly as they do to hosted APIs.

## 2. Register your TRUE economics (two measured numbers)

Your models have no list price, so Tokeymeter refuses to invent one: until
you register a rate, any USD shown is explicitly flagged
(`savings_report()["pricing"]["all_priced"] == False`). Registering takes
the two numbers your own ops can measure:

```python
import tokeymeter

d = tokeymeter.register_selfhost_pricing(
    "kimi-k2",                        # prefix covers kimi-k2-instruct, etc.
    gpu_hour_rate_usd=2.10,           # your amortized $/GPU-hour (hw+power+ops)
    measured_tokens_per_second=1400,  # sustained throughput of your serving stack
)
print(d["usd_per_1m_tokens"])         # 0.4166...  — derivation included in d
```

The returned dict embeds the full derivation
(`gpu_hour_rate / (tps × 3600) × 1e6`) so every figure is checkable by hand.
Where do the inputs come from?

- **gpu_hour_rate_usd** — (hardware amortization + power + hosting/ops) ÷
  GPU-hours in the period; or simply your cloud on-demand GPU rate.
- **measured_tokens_per_second** — your serving stack's own metrics
  (e.g. vLLM's `/metrics` Prometheus endpoint reports generation throughput),
  or total tokens ÷ wall-clock seconds over a representative window.

## 3. Read savings in GPU-hours (the unit you can't dispute)

```python
cap = tokeymeter.capacity_report(
    measured_tokens_per_second=1400,
    gpu_hour_rate_usd=2.10,           # optional — omit for capacity-only
)
# {'saved_tokens_total': ..., 'gpu_seconds_reclaimed': ...,
#  'gpu_hours_reclaimed': ..., 'equivalent_usd': ..., 'derivation': ...}
```

Every cache/single-flight hit is inference your GPUs did **not** run:
capacity returned to the cluster, queue depth you didn't accumulate,
scale-out you deferred. `capacity_report` states it in GPU-seconds derived
purely from *your* measured throughput — USD appears only if you supply
your own rate. Nothing is assumed; the derivation ships in the result.

## 4. See the whole story in one command

```bash
tokeymeter demo
```

Runs offline, keyless, fileless: (1) an unpriced model's USD gets *flagged*,
not laundered; (2) your true rate is derived with the math shown; (3) the
same workload reports honest USD **and** GPU-hours reclaimed. Act 4 —
allocated-vs-observed GPU reconciliation (utilization %, idle cost, GPUs
reclaimable) — lives in the TokeNet control plane for organization-wide
truth.

## FAQ

**Latency?** A cache hit returns in ~0.17 ms (p50, measured — run
`tokeymeter doctor`). Under load, hits also protect your p95 by skipping
the inference queue entirely.

**Does any of this leave my machine/perimeter?** No. The ledger is a local
file; the optional TokeNet emitter, if you later attach one, ships a
whitelisted content-blind projection (never prompt/response text) to a
control plane **you** host.

**Input vs output pricing?** On your own hardware a token-second is a
token-second at the throughput you measured, so both are priced identically
by default. If prefill and decode differ materially on your stack, measure
them separately and register split rates with
`tokeymeter.register_pricing(model, input_per_1m=..., output_per_1m=...)`.
