# AGENTS.md — instructions for a coding agent installing Tokeymeter

This file exists so a coding agent (Claude Code, Codex, Cursor) can add
Tokeymeter to a repository correctly without a human reading the docs first.

Point your coding agent at this file and say:

> Read AGENTS.md and add Tokeymeter task boundaries to my agent entry points.

---

## What Tokeymeter does

It stops an AI agent that has stopped getting anywhere.

An agent that hits a broken tool retries, and each retry costs more than the
last because the conversation history grows. Nothing catches this: the bill
cannot (repeats are served from cache, so a 20-call loop can cost under two
cents), an iteration cap cannot (it counts calls, not results), and a trace
viewer only shows it afterwards.

Tokeymeter watches whether the responses coming back are still new. When they
stop being new while the input keeps growing, it halts the task and says so.

---

## The only rule that matters when installing

**Wrap the AGENT ENTRY POINT — the function that handles one whole unit of work.
Never wrap individual model calls.**

One unit of work means one ticket, one document, one job, one user request.
Everything the agent does for that unit belongs inside a single `task()` block,
because the whole point is measuring progress ACROSS the calls in one unit. A
boundary drawn around a single call measures nothing.

If you are unsure whether something is an entry point, ask: *would a human call
this "one job"?* If yes, that is the boundary.

---

## Step 1 — install

```bash
pip install tokeymeter
```

## Step 2 — wrap the provider client once, at construction

```python
# OpenAI
from tokeymeter.engines.execution.integrations import openai as tokeymeter_openai
client = tokeymeter_openai.wrap(OpenAI())

# Anthropic
from tokeymeter.engines.execution.integrations import anthropic as tokeymeter_anthropic
client = tokeymeter_anthropic.wrap(Anthropic())
```

Do this once where the client is created. Do not wrap it repeatedly, and do not
wrap it inside a loop.

For any other provider (Bedrock, Gemini, a local server, a raw HTTP call), wrap
the function that makes the call:

```python
from tokeymeter.engines.execution.integrations.universal import meter

invoke = meter(
    bedrock.invoke_model,
    model="claude-sonnet",
    text_fn=lambda a, k: k["body"]["messages"][-1]["content"],   # the prompt
    usage_fn=lambda r: (r["usage"]["input_tokens"],
                        r["usage"]["output_tokens"]),            # the counts
)
```

## Step 3 — mark each unit of work

```python
import tokeymeter

with tokeymeter.task(f"ticket-{ticket.id}", agent="support"):
    agent.run(ticket)
```

- `task_id` identifies THIS unit — a ticket id, job id, run id. Never content.
- `agent` names the KIND of work — `"support"`, `"extract"`, `"research"`.
  Use the same value for every task of that kind; it is what reports group by.

**Add no ceilings yet.** Recording first is deliberate: it is impossible for
this to break anything, and the numbers it collects are what choose the
thresholds in step 5.

## Step 4 — confirm it works

```bash
python -m tokeymeter agents
```

Every wrapped agent should appear with a cost per task and a progress score.
If an agent is missing, its entry point was not wrapped.

If progress shows `-` instead of a number, responses could not be measured.
That happens when the function returns a provider object this node does not
recognise; pass an extractor:

```python
@tokeymeter.cache(model="gpt-4o",
                  extract_response_text=lambda r: r.choices[0].message.content)
```

## Step 5 — turn on ceilings, using the recommended values

```bash
python -m tokeymeter agents --suggest
```

This replays candidate thresholds against the traffic already recorded and
reports, for each one, how many stalled tasks it would have caught and how many
HEALTHY tasks it would also have stopped. Use the values it gives. Do not invent
numbers — a ceiling that stops working tasks gets switched off, and then it
protects nothing.

Then apply them:

```python
with tokeymeter.task(f"ticket-{ticket.id}", agent="support",
                     stall_window=8, enforce=True):
    agent.run(ticket)
```

And handle the halt where the work is dispatched:

```python
except tokeymeter.TaskStalled as exc:
    ticket.flag_for_human(reason=str(exc))
```

---

## Rules for the coding agent doing this work

1. **Do not wrap individual model calls.** One `task()` per unit of work.
2. **Do not add `enforce=True` in the first change.** Record first, run real
   traffic, then use `--suggest`. Enforcing on a guessed threshold is the one
   way this change can cause harm.
3. **Do not invent threshold values.** If asked for a number before there is
   history, say there is not enough data yet.
4. **Do not put content in `task_id` or `agent`.** They are labels and they are
   written to the ledger. A prompt, a customer name or a document body must
   never appear in either.
5. **Do not wrap anything in a `finally` or retry helper.** The boundary belongs
   at the entry point, not around the retry.
6. **Do not remove or modify existing observability.** Tokeymeter is not a
   replacement for a trace viewer — it tells you WHICH task went wrong, and the
   trace viewer tells you why. Keep both.
7. **If the codebase makes single-shot calls only** — one prompt, one answer, no
   loop — say so and stop. Stall detection has nothing to measure, and
   installing it would add a dependency for no benefit.

---

## What this does not do

- It cannot instrument Claude Code, Cursor or Codex. Those make their own calls
  from their own process; there is no call site in this repository to wrap.
- It catches agents that repeat themselves. An agent that is genuinely
  exploring — a different search each time — looks productive to it. Use
  `max_calls` or `envelope` for that case.
- It never reads prompts or responses. Both are hashed. Nothing leaves the
  process.

---

## Verify the change

```bash
python -m tokeymeter agents        # every wrapped agent appears
python -m tokeymeter watch         # live view, run in a second terminal
python -m tokeymeter firstrun      # offline demo, no key needed
```

If `tokeymeter agents` reports spend that belongs to no task, an entry point
was missed.
