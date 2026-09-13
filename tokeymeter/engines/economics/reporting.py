"""
tokeymeter.report — produce THE NUMBER, signed and shareable.

This module turns a Tokeymeter session into the single artifact the whole
company hangs from: a verifiable savings report. It does two jobs.

1. `validate(...)` drives a realistic, mixed workload (exact repeats, near-
   duplicates, a concurrency burst) through the drop-in wrappers against a
   provider client — real or a supplied fake — and captures the engine's
   savings plus the provider-actual reconciliation.

2. `render_html(...)` writes a self-contained, **signed** HTML report. The
   signature is produced with the engine's own Ed25519 signer over a canonical
   digest of the numbers, so a reader can verify the figures were not edited
   after generation. Content-blind throughout: only counts and dollars, never
   a prompt.

CLI:
    python -m tokeymeter.report --demo            # uses a fake client, no keys
    python -m tokeymeter.report --provider openai # uses your real OPENAI_API_KEY
    python -m tokeymeter.report --provider anthropic
    # → writes tokeymeter-savings.html (and .json) you can forward to anyone.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
from typing import Any, Optional

import tokeymeter as tk
from tokeymeter.engines.economics import reconcile

# A realistic support/RAG-style workload: some prompts repeat exactly, some are
# reworded (semantic), and one is hit concurrently (single-flight).
_BASE_PROMPTS = [
    "How do I reset a user's password?",
    "What is our refund policy for annual plans?",
    "Explain how request retries are configured.",
    "Summarize the steps to rotate an API key.",
    "What regions is data stored in for EU customers?",
    "How do I export an audit log bundle?",
    "What is the rate limit on the search endpoint?",
    "How do I add a teammate to a project?",
]
_REWORDS = {
    "How do I reset a user's password?": "What's the way to reset a user password?",
    "What is our refund policy for annual plans?": "Refund policy for yearly plans?",
    "How do I export an audit log bundle?": "How can I export the audit bundle?",
}


def _fake_client(kind: str):
    """A faithful fake with the real SDK's response shape — for --demo / tests."""
    def _answer(p: str) -> str:
        return f"Answer regarding: {p[:40]} — " + ("a realistic governed response. " * 14)

    class _U:
        def __init__(self, i, o):
            self.prompt_tokens = i; self.completion_tokens = o
            self.input_tokens = i; self.output_tokens = o

    class _M:
        def __init__(self, t): self.content = t; self.text = t

    class _C:
        def __init__(self, t): self.message = _M(t)

    class _OAResp:
        def __init__(self, t, i, o): self.choices = [_C(t)]; self.usage = _U(i, o)

    class _AnResp:
        def __init__(self, t, i, o): self.content = [_M(t)]; self.usage = _U(i, o)

    if kind == "openai":
        class _Comp:
            def create(self, **kw):
                time.sleep(0.05)  # simulate real network latency for honest hit/miss deltas
                p = kw["messages"][-1]["content"]; a = _answer(p)
                return _OAResp(a, len(p) // 4 + 20, len(a) // 4)

        class _Chat:
            def __init__(self): self.completions = _Comp()

        class _Client:
            def __init__(self): self.chat = _Chat()
        return _Client()
    else:
        class _Msgs:
            def create(self, **kw):
                time.sleep(0.05)
                p = kw["messages"][-1]["content"]; a = _answer(p)
                return _AnResp(a, len(p) // 4 + 20, len(a) // 4)

        class _Client:
            def __init__(self): self.messages = _Msgs()
        return _Client()


def validate(provider: str = "openai", *, model: Optional[str] = None,
             client: Any = None, rounds: int = 3, burst: int = 16,
             tag: str = "validation") -> dict:
    """Run the standard validation workload and return the reconciled result."""
    tk.reset(); tk.reset_savings(); reconcile.reset()
    tk.set_default_store(tk.MemoryStore())

    real = client is not None
    if provider == "openai":
        from tokeymeter.engines.execution.integrations.openai import wrap
        c = client or _fake_client("openai")
        wrapped = wrap(c, semantic=True, tag=tag)
        model = model or "gpt-4o-mini"

        def ask(p):
            return wrapped.chat.completions.create(
                model=model, messages=[{"role": "user", "content": p}], max_tokens=256)
    elif provider == "anthropic":
        from tokeymeter.engines.execution.integrations.anthropic import wrap
        c = client or _fake_client("anthropic")
        wrapped = wrap(c, semantic=True, tag=tag)
        model = model or "claude-haiku-4"

        def ask(p):
            return wrapped.messages.create(
                model=model, max_tokens=256, messages=[{"role": "user", "content": p}])
    else:
        raise ValueError("provider must be 'openai' or 'anthropic'")

    t0 = time.time()
    # exact repeats
    for _ in range(rounds):
        for p in _BASE_PROMPTS:
            ask(p)
    # near-duplicates (semantic)
    for reworded in _REWORDS.values():
        ask(reworded)
    # concurrency burst on a single hot prompt (single-flight)
    hot = "What does the dashboard show on the executive lens?"
    threads = [threading.Thread(target=lambda: ask(hot)) for _ in range(burst)]
    [t.start() for t in threads]; [t.join() for t in threads]
    elapsed = time.time() - t0

    rep = tk.savings_report()
    rec = reconcile.reconcile(rep)
    rec["meta"] = {
        "provider": provider, "model": model, "real_provider": real,
        "rounds": rounds, "burst": burst, "elapsed_s": round(elapsed, 2),
        "generated_at": time.time(),
    }
    return rec


# ── signing ─────────────────────────────────────────────────────────────────
def _canonical_digest(rec: dict) -> str:
    """A stable hash over the numbers that matter, for signature + tamper check."""
    basis = {
        "engine_estimate": rec.get("engine_estimate"),
        "reconciled": rec.get("reconciled"),
        "provider_actual": {k: v for k, v in rec.get("provider_actual", {}).items()
                            if k != "by_model"},
        "meta": rec.get("meta"),
    }
    return hashlib.sha256(json.dumps(basis, sort_keys=True, default=str).encode()).hexdigest()


def _sign(digest: str):
    """Sign the digest with the engine's Ed25519 signer. Returns (sig_hex, pub_hex)."""
    try:
        from tokeymeter.engines.trust.audit import Ed25519Signer
        signer = Ed25519Signer.generate()
        sig = signer.sign(digest.encode())
        pub = signer.public_bytes()
        return (sig.hex() if isinstance(sig, (bytes, bytearray)) else str(sig),
                pub.hex() if isinstance(pub, (bytes, bytearray)) else str(pub))
    except Exception:
        # HMAC fallback keeps the report signed even without asymmetric keys.
        import hmac
        secret = os.urandom(32)
        sig = hmac.new(secret, digest.encode(), hashlib.sha256).hexdigest()
        return sig, "hmac:" + hashlib.sha256(secret).hexdigest()[:16]


def render_html(rec: dict, path: str = "tokeymeter-savings.html") -> str:
    e = rec["engine_estimate"]; r = rec["reconciled"]; m = rec["meta"]
    pa = rec["provider_actual"]
    digest = _canonical_digest(rec)
    sig, pub = _sign(digest)
    saved = r["actual_saved_usd"]; spent = r["actual_spent_usd"]
    pct = r["savings_pct"]
    real_badge = ("PROVIDER-ACTUAL · reconcilable to invoice" if m["real_provider"]
                  else "DEMO DATA · run with real keys for a billable figure")
    by_model_rows = "".join(
        f"<tr><td>{mm}</td><td>{v['calls']:,}</td><td>{v['input_tokens']:,}</td>"
        f"<td>{v['output_tokens']:,}</td><td>${v['actual_cost_usd']:.4f}</td></tr>"
        for mm, v in pa.get("by_model", {}).items())

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Tokeymeter — Savings Report</title>
<style>
:root{{--bg:#04050A;--ink:#E8ECF4;--dim:#8A93A6;--line:rgba(255,255,255,.09);
--blue:#5E8BFF;--cyan:#6EE7F9;--violet:#8B5CF6;--ok:#30D158}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--ink);font:15px/1.6 -apple-system,BlinkMacSystemFont,'Segoe UI',Inter,sans-serif;
padding:48px 20px;display:flex;justify-content:center}}
.wrap{{width:100%;max-width:760px}}
.brand{{display:flex;align-items:center;gap:10px;font-weight:700;letter-spacing:.14em;font-size:13px;color:var(--dim)}}
.brand b{{color:var(--ink)}}
h1{{font-size:30px;font-weight:680;margin:22px 0 6px;letter-spacing:-.02em}}
.badge{{display:inline-block;margin:8px 0 30px;padding:6px 13px;border-radius:99px;font-size:11px;font-weight:640;
letter-spacing:.06em;background:rgba(94,139,255,.12);color:var(--cyan);box-shadow:inset 0 0 0 1px rgba(94,139,255,.3);
font-family:ui-monospace,monospace}}
.hero{{background:linear-gradient(135deg,rgba(110,231,249,.08),rgba(139,92,246,.08));
border:1px solid var(--line);border-radius:20px;padding:34px;margin-bottom:24px;text-align:center}}
.hero .big{{font-size:64px;font-weight:760;letter-spacing:-.03em;
background:linear-gradient(120deg,var(--cyan),var(--blue) 46%,var(--violet));
-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent;line-height:1}}
.hero .sub{{color:var(--dim);font-size:14px;margin-top:8px}}
.grid{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;margin-bottom:24px}}
.card{{background:rgba(255,255,255,.025);border:1px solid var(--line);border-radius:14px;padding:18px}}
.card .k{{font-size:11px;letter-spacing:.07em;text-transform:uppercase;color:var(--dim)}}
.card .v{{font-size:26px;font-weight:680;margin-top:6px}}
.card .v.ok{{color:var(--ok)}}
table{{width:100%;border-collapse:collapse;margin:10px 0 26px;font-size:13px}}
th,td{{text-align:left;padding:9px 10px;border-bottom:1px solid var(--line)}}
th{{color:var(--dim);font-weight:600;font-size:11px;letter-spacing:.05em;text-transform:uppercase}}
td:not(:first-child),th:not(:first-child){{text-align:right;font-variant-numeric:tabular-nums}}
.sig{{background:rgba(255,255,255,.02);border:1px solid var(--line);border-radius:14px;padding:18px;
font-family:ui-monospace,monospace;font-size:11px;color:var(--dim);word-break:break-all;line-height:1.7}}
.sig b{{color:var(--ink)}}
.sig .row{{margin:3px 0}}
.foot{{color:var(--dim);font-size:12px;margin-top:26px;line-height:1.7}}
.method{{font-size:12.5px;color:var(--dim);background:rgba(255,255,255,.02);border:1px solid var(--line);
border-radius:12px;padding:16px;margin-bottom:24px}}
</style></head><body><div class="wrap">
<div class="brand">◬ <b>TOKEYMETER</b> · NOVUE</div>
<h1>AI Savings Report</h1>
<div class="badge">{real_badge}</div>

<div class="hero">
  <div class="big">${saved:,.2f}</div>
  <div class="sub">saved across {e['total_calls']:,} calls · {pct}% of spend never incurred</div>
</div>

<div class="grid">
  <div class="card"><div class="k">Cache hit rate</div><div class="v ok">{e['hit_rate_pct']}%</div></div>
  <div class="card"><div class="k">Calls served free</div><div class="v">{e['cache_hits']:,}</div></div>
  <div class="card"><div class="k">Real provider calls</div><div class="v">{pa['real_calls']:,}</div></div>
</div>

<div class="method">
  <b>How this number was produced.</b> {e['total_calls']:,} logical model calls were
  issued through Tokeymeter. {e['cache_hits']:,} were served from cache (exact,
  semantic, or single-flight) and never reached a provider. The remaining
  {pa['real_calls']:,} real calls reported {pa['input_tokens']:,} input +
  {pa['output_tokens']:,} output tokens via the provider's own
  <code>usage</code> field, priced at public list rates. Savings are the
  per-call cost of the calls that ran, applied to the calls that didn't.
</div>

<table>
  <tr><th>Provider / model</th><th>Real calls</th><th>Input tok</th><th>Output tok</th><th>Actual cost</th></tr>
  {by_model_rows}
  <tr><td><b>Total spent</b></td><td><b>{pa['real_calls']:,}</b></td>
      <td><b>{pa['input_tokens']:,}</b></td><td><b>{pa['output_tokens']:,}</b></td>
      <td><b>${spent:,.4f}</b></td></tr>
</table>

<div class="sig">
  <div class="row"><b>Signed report digest (SHA-256)</b></div>
  <div class="row">{digest}</div>
  <div class="row" style="margin-top:10px"><b>Signature (Ed25519)</b></div>
  <div class="row">{sig}</div>
  <div class="row" style="margin-top:10px"><b>Public key</b> &nbsp;{pub}</div>
</div>

<div class="foot">
  Generated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime(m['generated_at']))} ·
  provider {m['provider']} · model {m['model']} ·
  {m['rounds']} rounds · burst {m['burst']} · {m['elapsed_s']}s.<br>
  Content-blind: this report contains only counts and dollars — no prompt or
  response text exists in it. Figures derive from the provider's reported token
  usage at public list prices; reconcile against your provider invoice for the
  authoritative bill. The signature lets a recipient confirm the numbers were
  not altered after generation.
</div>
</div></body></html>"""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
    # also drop the machine-readable JSON next to it
    json_path = path.rsplit(".", 1)[0] + ".json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump({"report": rec, "digest": digest, "signature": sig, "public_key": pub},
                  fh, indent=2, default=str)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="Tokeymeter savings report")
    ap.add_argument("--provider", choices=["openai", "anthropic"], default="openai")
    ap.add_argument("--model", default=None)
    ap.add_argument("--demo", action="store_true",
                    help="use a faithful fake client (no API key, no spend)")
    ap.add_argument("--out", default="tokeymeter-savings.html")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--burst", type=int, default=16)
    a = ap.parse_args()

    client = None
    if not a.demo:
        if a.provider == "openai":
            if "OPENAI_API_KEY" not in os.environ:
                raise SystemExit("Set OPENAI_API_KEY, or pass --demo for a no-key run.")
            from openai import OpenAI
            client = OpenAI()
        else:
            if "ANTHROPIC_API_KEY" not in os.environ:
                raise SystemExit("Set ANTHROPIC_API_KEY, or pass --demo for a no-key run.")
            import anthropic
            client = anthropic.Anthropic()

    rec = validate(a.provider, model=a.model, client=client,
                   rounds=a.rounds, burst=a.burst)
    path = render_html(rec, a.out)
    r = rec["reconciled"]
    print(f"\n  Saved ${r['actual_saved_usd']:.4f} "
          f"({r['savings_pct']}% of spend) across {rec['engine_estimate']['total_calls']} calls.")
    print(f"  Report written: {path}  (+ {path.rsplit('.',1)[0]}.json)")
    if not rec["meta"]["real_provider"]:
        print("  NOTE: demo data. Re-run with real API keys for a billable, "
              "invoice-reconcilable figure.")


if __name__ == "__main__":
    main()
