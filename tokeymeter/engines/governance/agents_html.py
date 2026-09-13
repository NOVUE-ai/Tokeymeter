"""The report card as a single self-contained file.

WHY THIS EXISTS
---------------
`tokeymeter agents` prints to a terminal, and a terminal is where a finding goes
to die. The person who installs this is a developer; the person who decides
anything about it is their platform lead, and nothing travels between those two
except a link or an image.

So this writes ONE file. No server, no port, no account, no build step, and no
network request when it opens — an operator can drop it in Slack, attach it to a
ticket, or open it on a laptop with no Python on it.

WHAT IT DELIBERATELY IS NOT
---------------------------
Not a dashboard. A dashboard implies something running, something to log into,
something that goes stale. This is a snapshot with the moment it was taken
printed on it, which is the honest shape for a file that will be forwarded and
read three days later.

THE DESIGN, AND WHY IT IS NOT FOUR STAT CARDS
---------------------------------------------
The obvious treatment — a row of big numbers with small labels — says nothing a
reader could not get from the table below it. The concept this product exists
for is a SPECTRUM: agents sit somewhere between "every answer is new" and "the
same answer over and over". So the top of the page is that spectrum, drawn once:
a 0-to-1 progress axis with each agent placed at its median and every stalled
task ticked beneath it. One glance gives the shape of the estate, and the shape
is the point.

Everything else is deliberately quiet. One bold element, then discipline.

NO WEB FONTS, NO NETWORK, EVER
------------------------------
A remote font would make the file phone home the moment a stranger opened it,
which contradicts the one property that makes it safe to forward. Type is a
local stack, and hierarchy is carried by weight and scale instead. The logo is
inline SVG with no `xmlns` attribute — HTML5 parses inline SVG into the SVG
namespace on its own, and that attribute's value is a URL, which a test
correctly forbids anywhere in the document.

CONTENT-BLIND, AND VISIBLY SO
-----------------------------
It renders identifiers, counts and money — nothing else, because nothing else
exists in the ledger. That is stated in the footer rather than assumed, since
the whole point is that this file gets forwarded to people who did not install
it and have every right to ask what is in it.

EVERY VALUE IS ESCAPED
----------------------
Task ids and agent names come from the operator, and an operator who names an
agent `<script>` should get a page that says `<script>`, not one that runs it.
"""
from __future__ import annotations

import html
import os
import time
from typing import Any, Dict, List, Optional

__all__ = ["render_agents_html", "write_agents_html"]

# The NOVUE mark: three interlocking petals around a lens. Drawn rather than
# embedded so it stays a few hundred bytes and remains crisp at any size.
_LOGO = """
<svg class="mark" viewBox="-72 -72 144 144" role="img" aria-label="NOVUE">
  <defs>
    <linearGradient id="nv" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#0A6FFF"/>
      <stop offset="1" stop-color="#22D3F0"/>
    </linearGradient>
    <radialGradient id="nl" cx="50%" cy="50%" r="50%">
      <stop offset="0" stop-color="#7FE7FF"/>
      <stop offset="1" stop-color="#0A6FFF"/>
    </radialGradient>
  </defs>
  <g fill="none" stroke="url(#nv)" stroke-width="7" stroke-linejoin="round">
    <path d="M0,-62 C36,-26 36,14 0,34 C-36,14 -36,-26 0,-62 Z"/>
    <path d="M0,-62 C36,-26 36,14 0,34 C-36,14 -36,-26 0,-62 Z"
          transform="rotate(120)"/>
    <path d="M0,-62 C36,-26 36,14 0,34 C-36,14 -36,-26 0,-62 Z"
          transform="rotate(240)"/>
  </g>
  <circle r="21" fill="#F7FBFF"/>
  <circle r="14" fill="none" stroke="#0A6FFF" stroke-width="2.4"/>
  <circle r="6" fill="url(#nl)"/>
</svg>
"""

_CSS = """
:root{
  /* Straight off the mark: a blue-black field, one blue-to-cyan sweep. */
  --ink:#05070D; --panel:#0A0E15; --raise:#101724;
  --line:#161E2C; --line-lit:#26334A;
  --text:#EAEFF6; --dim:#A7B3C4; --muted:#76839A;
  --brand-a:#0A6FFF; --brand-b:#22D3F0; --lens:#7FE7FF;
  --stall:#FF6B5B; --watch:#F5B544; --ok:#3DDC97;
  --sweep:linear-gradient(90deg,var(--brand-a),var(--brand-b));
  --r:10px;
  /* A fourth-based scale: 12 / 13 / 15 / 20 / 27. */
  --t-xs:12px; --t-sm:13px; --t-base:15px; --t-lg:20px; --t-xl:27px;
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{
  margin:0; background:var(--ink); color:var(--text);
  font-family:"Inter var",Inter,"SF Pro Text","Segoe UI Variable Text",
              "Segoe UI",-apple-system,BlinkMacSystemFont,"Helvetica Neue",
              "Noto Sans",Roboto,Arial,sans-serif;
  font-size:var(--t-base); line-height:1.6; font-weight:400;
  font-variant-numeric:tabular-nums lining-nums;
  font-feature-settings:"cv05" 1,"ss01" 1;
  letter-spacing:-.006em;
  -webkit-font-smoothing:antialiased; -moz-osx-font-smoothing:grayscale;
}
/* Frosted glass needs something behind it or it is just a grey box. Two very
   soft washes in the mark's own blue and cyan, fixed so they do not scroll
   with the content. */
body::before{content:"";position:fixed;inset:0;pointer-events:none;z-index:0;
  background:
    radial-gradient(58% 46% at 12% -6%,rgba(10,111,255,.20),transparent 68%),
    radial-gradient(44% 38% at 96% 10%,rgba(34,211,240,.13),transparent 70%),
    radial-gradient(50% 40% at 70% 105%,rgba(10,111,255,.10),transparent 70%)}
.page{position:relative;z-index:1;max-width:980px;margin:0 auto;
      padding:60px 32px 80px}

/* The glass itself: a translucent fill, a real backdrop blur, and a specular
   hairline along the top edge where the light would catch. */
.glass{
  background:linear-gradient(180deg,rgba(255,255,255,.055),
                                    rgba(255,255,255,.016));
  backdrop-filter:blur(24px) saturate(155%);
  -webkit-backdrop-filter:blur(24px) saturate(155%);
  border:1px solid rgba(255,255,255,.085);
  box-shadow:0 26px 60px -34px rgba(0,0,0,.95),
             inset 0 1px 0 rgba(255,255,255,.12)}
@supports not ((backdrop-filter:blur(1px)) or (-webkit-backdrop-filter:blur(1px))){
  .glass{background:var(--panel);border-color:var(--line)}
}

/* masthead */
.top{display:flex;align-items:center;gap:15px;padding-bottom:22px;
     border-bottom:1px solid var(--line)}
.mark{width:36px;height:36px;flex:none;
      filter:drop-shadow(0 0 16px rgba(34,211,240,.3))}
.word{font-size:var(--t-sm);letter-spacing:.44em;font-weight:300;
      color:var(--dim);padding-top:1px}
.top .spacer{flex:1}
.taken{font-size:var(--t-xs);color:var(--muted);text-align:right;
       line-height:1.5}
h1{font-size:var(--t-xl);font-weight:600;letter-spacing:-.028em;
   line-height:1.15;margin:38px 0 8px}
.lede{color:var(--dim);margin:0 0 34px;max-width:58ch;font-size:16px;
      line-height:1.6}

/* the spectrum: the one bold element */
.spectrum{position:relative;overflow:hidden;border-radius:16px;
          padding:26px 30px 20px}
.spectrum h2{font-size:var(--t-sm);font-weight:500;color:var(--dim);
             margin:0 0 24px;letter-spacing:0}
.axis{position:relative;height:150px;margin:0 8px}
.rail-dim{position:absolute;left:0;right:0;top:66px;height:2px;
          background:var(--line);border-radius:2px}
.rail{position:absolute;left:0;right:0;top:66px;height:2px;border-radius:2px;
      background:linear-gradient(90deg,var(--stall),var(--watch) 32%,
                 var(--brand-a) 70%,var(--brand-b));
      transform-origin:left center;animation:draw .9s cubic-bezier(.2,.7,.3,1)}
@keyframes draw{from{transform:scaleX(0);opacity:0}to{transform:scaleX(1);opacity:1}}

.node{position:absolute;transform:translateX(-50%);text-align:center;
      width:146px;top:6px}
.node b{display:block;font-size:var(--t-sm);font-weight:500;
        white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
        transition:color .18s ease}
.node s{display:block;text-decoration:none;font-size:var(--t-xs);
        color:var(--muted);margin-top:2px}
.stem{position:absolute;left:50%;width:1px;background:var(--line-lit);
      transform:translateX(-50%);top:44px;height:20px;
      transition:background .18s ease}
.dot{position:absolute;top:60px;left:50%;width:14px;height:14px;
     border-radius:50%;border:3px solid var(--ink);
     box-shadow:0 0 0 1px rgba(255,255,255,.22);z-index:2;
     transform:translateX(-50%);
     transition:transform .2s cubic-bezier(.2,.7,.3,1),box-shadow .2s ease}
.node:hover .dot{transform:translateX(-50%) scale(1.45);
                 box-shadow:0 0 0 1px rgba(255,255,255,.5),
                            0 0 22px rgba(255,255,255,.22)}
.node:hover b{color:var(--lens)}
.node:hover .stem{background:var(--brand-b)}

.cluster{position:absolute;top:76px;transform:translateX(-50%);
         text-align:center;width:120px}
.cluster i{display:block;width:3px;height:18px;border-radius:2px;margin:0 auto;
           background:var(--stall);box-shadow:0 0 12px rgba(255,107,91,.5)}
.cluster em{display:block;font-style:normal;font-size:var(--t-xs);
            color:var(--stall);margin-top:6px;white-space:nowrap;
            letter-spacing:.01em}
.ends{display:flex;justify-content:space-between;margin:4px 8px 0;
      font-size:var(--t-xs);color:var(--muted)}
.legend{list-style:none;display:flex;flex-wrap:wrap;gap:9px 24px;
        margin:22px 8px 0;padding:0}
.legend li{display:flex;align-items:center;gap:9px;font-size:var(--t-sm)}
.legend em{font-style:normal;color:var(--muted);font-size:var(--t-xs)}
.key{width:9px;height:9px;border-radius:50%;flex:none}
.spectrum.flat .axis{height:38px}
.spectrum.flat .rail,.spectrum.flat .rail-dim{top:16px}
.spectrum.flat .dot{top:10px}
.spectrum.flat .cluster{top:26px}
.spectrum.flat .cluster em{display:none}

/* table */
table{width:100%;border-collapse:collapse;margin-top:40px}
caption{text-align:left;font-size:var(--t-sm);color:var(--dim);
        padding-bottom:14px}
th{text-align:left;font-size:var(--t-xs);font-weight:500;color:var(--muted);
   padding:0 14px 9px 14px;border-bottom:1px solid var(--line);
   letter-spacing:.005em}
td{padding:14px;border-bottom:1px solid var(--line);font-size:14px;
   transition:background .16s ease}
tr:last-child td{border-bottom:none}
/* A right-aligned column with no left padding collides with the cell before
   it: the headers rendered as "CallsProgress" and the progress bar ran into
   the calls figure. */
th:first-child,td:first-child{padding-left:0}
th:last-child,td:last-child{padding-right:0}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
td.name{font-weight:500;position:relative}
tbody tr:hover td{background:rgba(34,211,240,.045)}
tbody tr:hover td.name{color:var(--lens)}
td.name::before{content:"";position:absolute;left:-14px;top:8px;bottom:8px;
  width:2px;border-radius:2px;background:var(--sweep);opacity:0;
  transition:opacity .16s ease}
tbody tr:hover td.name::before{opacity:1}
.meter{display:inline-flex;align-items:center;gap:10px}
.bar{width:66px;height:5px;border-radius:3px;background:var(--line);
     overflow:hidden;flex:none}
.bar i{display:block;height:100%;border-radius:3px}
.pill{display:inline-block;min-width:26px;padding:2px 10px;border-radius:999px;
      font-size:var(--t-sm);font-weight:500;
      background:rgba(255,107,91,.14);color:var(--stall)}
.zero{color:var(--muted)}

/* stalled — collapsible, because a long list should not bury the footer */
.stalls{margin-top:40px}
.stalls summary{cursor:pointer;list-style:none;display:flex;
  align-items:baseline;gap:10px;font-size:var(--t-base);font-weight:600;
  padding:2px 0;border-radius:6px;transition:color .16s ease}
.stalls summary::-webkit-details-marker{display:none}
.stalls summary em{font-style:normal;font-weight:500;font-size:var(--t-sm);
  color:var(--stall);background:rgba(255,107,91,.13);
  padding:2px 10px;border-radius:999px;margin-left:2px}
.stalls summary::after{content:"";width:6px;height:6px;margin-left:2px;
  border-right:1.5px solid var(--muted);border-bottom:1.5px solid var(--muted);
  transform:rotate(45deg) translateY(-2px);
  transition:transform .2s ease,border-color .16s ease}
.stalls[open] summary::after{transform:rotate(225deg) translateY(-2px)}
.stalls summary:hover{color:var(--lens)}
.stalls summary:hover::after{border-color:var(--lens)}
.stalls summary:focus-visible{outline:2px solid var(--brand-b);
  outline-offset:4px}
.stalls .hint{margin:6px 0 16px;color:var(--muted);font-size:var(--t-sm);
              max-width:62ch}
.stalls ol{list-style:none;padding:0;margin:0}
.stalls li{display:flex;flex-wrap:wrap;align-items:baseline;gap:8px 20px;
  padding:14px 18px;margin-bottom:8px;border-radius:12px;
  background:linear-gradient(180deg,rgba(255,255,255,.042),
                                    rgba(255,255,255,.012));
  backdrop-filter:blur(16px) saturate(140%);
  -webkit-backdrop-filter:blur(16px) saturate(140%);
  border:1px solid rgba(255,255,255,.07);
  border-left:2px solid var(--stall);
  box-shadow:inset 0 1px 0 rgba(255,255,255,.08);
  transition:background .18s ease,border-color .18s ease,transform .18s ease}
.stalls li:hover{background:linear-gradient(180deg,rgba(255,255,255,.075),
                                                   rgba(255,255,255,.025));
  border-color:rgba(255,255,255,.13);border-left-color:var(--stall);
  transform:translateX(2px)}
@supports not ((backdrop-filter:blur(1px)) or (-webkit-backdrop-filter:blur(1px))){
  .stalls li{background:var(--raise);border-color:var(--line);
             border-left-color:var(--stall)}
}
.stalls li b{font-weight:500;font-size:14px}
.stalls li em{font-style:normal;color:var(--muted);font-size:var(--t-sm)}
.stalls li span{color:var(--dim);font-size:var(--t-sm)}

/* footer */
.note{margin-top:48px;padding-top:24px;border-top:1px solid var(--line);
      color:var(--muted);font-size:var(--t-sm);line-height:1.75;max-width:70ch}
.note strong{color:var(--dim);font-weight:500}
.note code{background:rgba(255,255,255,.055);
           border:1px solid rgba(255,255,255,.09);
           backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);
           padding:1px 6px;border-radius:5px;font-size:var(--t-xs);
           font-family:ui-monospace,"SF Mono",SFMono-Regular,Menlo,Consolas,
                       "Liberation Mono",monospace}
.note p{margin:0 0 13px}
.empty{padding:32px 0;color:var(--muted)}
@media (max-width:680px){
  .page{padding:38px 20px 60px}
  h1{font-size:22px}
  .node{width:104px}
  .axis{height:150px}
}
@media (prefers-reduced-motion:reduce){
  *{animation:none!important;transition:none!important}
}
"""


def _e(value: Any) -> str:
    """Escape anything before it reaches the page. An operator who names an
    agent `<script>` gets a page that SAYS `<script>`."""
    return html.escape("" if value is None else str(value), quote=True)


def _num(value: Any, fmt: str = "{:.4f}", prefix: str = "") -> str:
    try:
        return prefix + fmt.format(float(value))
    except (TypeError, ValueError):
        return "&mdash;"


def _bar_colour(progress: float) -> str:
    """Healthy work carries the mark's own blue-to-cyan sweep; anything that
    needs attention breaks out of the brand palette on purpose, so a problem
    never reads as decoration."""
    if progress > 0.66:
        return "var(--sweep)"
    if progress > 0.25:
        return "var(--watch)"
    return "var(--stall)"


def _spectrum(agents: List[Dict[str, Any]]) -> str:
    """Every agent placed on a 0-to-1 progress axis, stalled tasks ticked
    beneath. The concept this product exists for, drawn once.

    TWO MODES, because a horizontal axis genuinely cannot label five agents
    that all score 1.00 — and an estate where everything is healthy is the
    normal case, not an edge one. When the agents are spread out the labels sit
    on the axis where they belong; when they cluster, the dots stay where they
    are and the names move to a legend underneath. Overlapping text would look
    broken and misreport the data at the same time.
    """
    def pos(fraction: float) -> float:
        # Inset: a label centred at 100% would hang off the panel, and an agent
        # scoring exactly 1.00 is the common case.
        return 10.0 + fraction * 80.0

    placed = []
    for a in agents:
        p_val = a.get("median_progress")
        if p_val is None:
            continue
        try:
            pf = max(0.0, min(1.0, float(p_val)))
        except (TypeError, ValueError):
            continue
        placed.append((pf, a))
    if not placed:
        return ""
    placed.sort(key=lambda x: x[0])

    # Stalled tasks cluster hard — six tasks stuck the same way all score the
    # same progress, and six hairlines drawn on top of each other look like
    # one. Group them and show the count, because "this agent is bimodal" is
    # exactly what the axis exists to reveal: a median of 1.00 sitting next to
    # six stalls is not a contradiction, it is two populations.
    # COUNT THE AGENT'S REAL STALLS, not the listed sample. `worst_stalls` is
    # truncated for display, so counting its entries printed "5 stalled" on the
    # axis beside a "6" in the table — two numbers on one page disagreeing,
    # which is worse than no number. Each agent's full stalled_tasks total goes
    # to the bucket where its stalls actually sit.
    buckets: Dict[int, int] = {}
    for _, a in placed:
        total = int(a.get("stalled_tasks") or 0)
        if not total:
            continue
        spots = []
        for st in (a.get("worst_stalls") or []):
            if not isinstance(st, dict):
                continue
            try:
                spots.append(max(0.0, min(1.0, float(st.get("progress")))))
            except (TypeError, ValueError):
                continue
        if not spots:
            continue
        spots.sort()
        at = round(pos(spots[len(spots) // 2]))
        buckets[at] = buckets.get(at, 0) + total
    ticks = []
    for left, count in sorted(buckets.items()):
        label = (f'<em>{count} stalled</em>' if count > 1 else "<em>stalled</em>")
        ticks.append(f'<div class="cluster" style="left:{left}%">'
                     f'<i></i>{label}</div>')

    # A label is about 104px on a ~850px panel, so 13% is roughly one width.
    # 146px labels on a ~850px panel: about 18% apart before they touch.
    crowded = any(pos(placed[i + 1][0]) - pos(placed[i][0]) < 18.0
                  for i in range(len(placed) - 1))

    marks, legend = [], []
    for pf, a in placed:
        left = pos(pf)
        # COLOUR MEANS ONE THING: how much progress. Colouring the dot by
        # "has stalls" instead put the same agent in cyan on the axis and
        # amber in the table, which reads as a contradiction even though both
        # rules were individually right. Stalls are carried by the cluster
        # mark below and the pill in the table.
        tone = _bar_colour(pf)
        name = _e(a.get("agent") or "&mdash;")
        tasks = int(a.get("tasks") or 0)
        if crowded:
            marks.append(f'<i class="dot" style="left:{left:.1f}%;'
                         f'background:{tone}"></i>')
            legend.append(
                f'<li><i class="key" style="background:{tone}"></i>{name}'
                f'<em>{pf:.2f} &middot; {tasks} tasks</em></li>')
        else:
            marks.append(
                f'<div class="node" style="left:{left:.1f}%">'
                f'<b>{name}</b><s>{tasks} tasks</s>'
                f'<i class="stem"></i>'
                f'<i class="dot" style="background:{tone}"></i></div>')

    legend_html = (f'<ul class="legend">{"".join(legend)}</ul>'
                   if legend else "")
    return f"""
<section class="spectrum glass{' flat' if crowded else ''}">
  <h2>Where each agent sits between repeating itself and making progress</h2>
  <div class="axis">
    <div class="rail-dim"></div><div class="rail"></div>
    {''.join(marks)}{''.join(ticks)}
  </div>
  <div class="ends"><span>0.00 &nbsp;the same answer, again</span>
    <span>a new answer every turn&nbsp; 1.00</span></div>
  {legend_html}
</section>"""


def render_agents_html(report: Dict[str, Any],
                       *, generated_at: Optional[float] = None) -> str:
    """A complete HTML document for an agent report. Never raises on odd data:
    a missing field renders as a dash, because a report that fails to render is
    worth less than one with a gap in it."""
    agents: List[Dict[str, Any]] = [a for a in (report.get("agents") or [])
                                    if isinstance(a, dict)]
    when = time.strftime("%d %B %Y, %H:%M %Z",
                         time.localtime(generated_at or time.time()))
    total_tasks = report.get("total_tasks") or sum(
        int(a.get("tasks") or 0) for a in agents)
    stalled_total = sum(int(a.get("stalled_tasks") or 0) for a in agents)
    spend = sum(float(a.get("spend_usd") or 0.0) for a in agents)

    # Two decimals reads "$0.00" on a real gpt-4o-mini run that spent $0.0038.
    # Show enough places for the number to exist.
    def _money(v: float) -> str:
        # Two decimals reads "$0.00" on a real gpt-4o-mini run that spent
        # $0.0038. Show enough places for the number to exist.
        return f"${v:,.2f}" if v >= 1 else f"${v:.4f}"

    # THE number a platform lead acts on, and it was missing entirely: not what
    # you spent, but what you spent on tasks that produced nothing.
    wasted = sum(float(a.get("stalled_spend_usd") or 0.0) for a in agents)
    share = (wasted / spend * 100) if spend > 0 else 0.0

    lede = (f"{total_tasks} tasks across {len(agents)} "
            f"agent{'' if len(agents) == 1 else 's'}, {_money(spend)} spent.")
    if stalled_total and wasted > 0:
        lede += (f" {_money(wasted)} of that &mdash; {share:.0f}% &mdash; went "
                 f"to {stalled_total} task"
                 f"{'' if stalled_total == 1 else 's'} that stopped making "
                 f"progress.")
    elif stalled_total:
        lede += (f" {stalled_total} stopped making progress while still "
                 f"costing money.")
    elif agents:
        lede += " Nothing stopped making progress in this window."

    rows = []
    for a in sorted(agents, key=lambda x: -(float(x.get("spend_usd") or 0))):
        p = a.get("median_progress")
        if p is None:
            meter = '<span class="zero">not scored</span>'
        else:
            pf = max(0.0, min(1.0, float(p)))
            meter = (f'<span class="meter"><span class="bar">'
                     f'<i style="width:{pf * 100:.0f}%;'
                     f'background:{_bar_colour(pf)}"></i></span>'
                     f'{pf:.2f}</span>')
        stalled = int(a.get("stalled_tasks") or 0)
        rows.append(
            "<tr>"
            f'<td class="name">{_e(a.get("agent") or "&mdash;")}</td>'
            f'<td class="n">{int(a.get("tasks") or 0)}</td>'
            f'<td class="n">{_num(a.get("median_cost_per_task_usd"), prefix="$")}</td>'
            f'<td class="n">{_num(a.get("p95_cost_per_task_usd"), prefix="$")}</td>'
            f'<td class="n">{_num(a.get("median_calls_per_task"), "{:.1f}")}</td>'
            f"<td>{meter}</td>"
            f'<td class="n">'
            + (f'<span class="pill">{stalled}</span>' if stalled
               else '<span class="zero">0</span>')
            + "</td></tr>")

    if rows:
        table = f"""
<table>
  <caption>Cost and progress per agent, highest spend first</caption>
  <thead><tr><th>Agent</th><th class="n">Tasks</th><th class="n">Median $</th>
  <th class="n">p95 $</th><th class="n">Calls</th><th>Progress</th>
  <th class="n">Stalled</th></tr></thead>
  <tbody>{''.join(rows)}</tbody>
</table>"""
    else:
        table = ('<p class="empty">No agent tasks recorded yet. Wrap an agent '
                 'entry point with <code>tokeymeter.task(...)</code> and run '
                 'some traffic.</p>')

    detail = []
    for a in agents:
        for s in (a.get("worst_stalls") or []):
            if isinstance(s, dict):
                detail.append((a.get("agent"), s))
    stalls_html = ""
    if detail:
        items = []
        for agent, s in detail[:20]:
            growth = s.get("input_growth")
            items.append(
                "<li>"
                f'<b>{_e(s.get("task_id"))}</b>'
                f"<em>{_e(agent)}</em>"
                f'<span>progress {float(s.get("progress") or 0):.2f}</span>'
                + (f"<span>input +{float(growth):.0%}</span>"
                   if growth is not None else "")
                + f'<span>{int(s.get("calls") or 0)} calls</span>'
                f'<span>${float(s.get("spend_usd") or 0):.4f}</span></li>')
        shown = len(items)
        # <details> rather than a script: this file must contain no JavaScript,
        # and a reader who has twenty stalled tasks should be able to fold them
        # away to reach the footer. Open by default, because the person this
        # gets forwarded to should see the list without being asked to click.
        stalls_html = f"""
<details class="stalls" open>
  <summary>{stalled_total} task{'' if stalled_total == 1 else 's'} stopped making progress<em>{_money(wasted)}</em></summary>
  <p class="hint">Each returned the same answer while its input kept growing. Open these in your trace viewer first{f' &mdash; showing the {shown} worst' if shown < stalled_total else ''}.</p>
  <ol>{''.join(items)}</ol>
</details>"""

    th = report.get("thresholds") or {}
    low_p = float(th.get("low_progress") or 0.25)
    growth_p = float(th.get("input_growth") or 0.25)
    min_calls = int(th.get("min_calls_scored") or 4)

    # Coverage is a trust statement, not a footnote. A reader deciding whether
    # to act on this needs to know how much of the traffic it could actually
    # measure — a report that quietly ignores half the calls is worse than one
    # that says so.
    scored = sum(int(a.get("responses_scored") or 0) for a in agents)
    executed = sum(int(a.get("responses_executed") or 0) for a in agents)
    malformed = int(report.get("excluded_malformed_records") or 0)
    if executed:
        pct = scored / executed * 100
        coverage_line = (
            f"{scored:,} of {executed:,} executed responses could be scored "
            f"({pct:.0f}%).")
        if pct < 99.5:
            coverage_line += (" The rest returned something this node could "
                              "not read as text, so their tasks are not "
                              "judged on progress.")
    else:
        coverage_line = "No executed responses in this window."
    if malformed:
        coverage_line += (f" {malformed:,} malformed record"
                          f"{'' if malformed == 1 else 's'} were skipped.")

    untasked = float(report.get("untasked_spend_usd") or 0.0)
    untasked_note = (
        f"<p><strong>Not everything is wrapped.</strong> {_money(untasked)} of "
        f"spend belongs to no task, so it appears in no row above. Those entry "
        f"points have no <code>tokeymeter.task(...)</code> boundary yet.</p>"
        if untasked > 0 else "")

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Agent report &mdash; {_e(when)}</title><style>{_CSS}</style></head>
<body><div class="page">

<header class="top">
  {_LOGO}
  <div class="word">NOVUE</div>
  <div class="spacer"></div>
  <div class="taken">Snapshot taken<br>{_e(when)}</div>
</header>

<h1>Agent report</h1>
<p class="lede">{lede}</p>

{_spectrum(agents)}
{table}
{stalls_html}

<div class="note">
  <p><strong>Progress</strong> is distinct responses divided by executed calls.
  An agent that is working returns new information; a stuck one returns the same
  answer again. Low progress with <strong>flat</strong> input is normal &mdash;
  that is a classifier. Low progress with <strong>growing</strong> input is a
  stall: paying more to learn nothing new.</p>
  <p>Responses are hashed, never read. This file holds identifiers, counts and
  money only &mdash; no prompt or response text exists in the ledger it was
  built from, and opening it makes no network request.</p>
  <p><strong>How a stall was decided.</strong> A task is flagged only when its
  progress falls to {low_p:.2f} or below <em>and</em> its input grew by at least
  {growth_p:.0%}, over at least {min_calls} scored responses. Low progress with
  flat input is legitimate repetitive work and is never flagged.</p>
  <p><strong>What this report can see.</strong> {coverage_line}</p>
  {untasked_note}
  <p>Generated by <code>tokeymeter agents --html</code>. This is a snapshot, not
  a live view; re-run it to refresh.</p>
</div>

</div></body></html>"""


def write_agents_html(report: Dict[str, Any],
                      path: str = "tokeymeter-agents.html",
                      *, generated_at: Optional[float] = None) -> str:
    """Write the report and return the absolute path.

    UTF-8 explicitly: this is written on one machine and opened on another, and
    a file that renders as mojibake on a colleague's laptop has failed at the
    one job it has.
    """
    doc = render_agents_html(report, generated_at=generated_at)
    directory = os.path.dirname(os.path.abspath(path))
    if directory and not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc)
    return os.path.abspath(path)
