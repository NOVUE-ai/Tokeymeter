"""Compliance context and enforcement — what a call is ALLOWED to do.

WHY THIS IS A SEPARATE CONCERN
------------------------------
Every rule the engine could express until now answered "how much": envelopes,
call limits, stall windows. None answered "what is permitted". So a compliance
officer could approve this product in an afternoon — content-blind, no
subprocessor, an audit chain — and then could not author a single thing in it.

This module supplies the missing half: two declared facts about a request, and
the three actions a policy can take on them.

    - when: {data_class: PHI}
      then: {only: [gpt-4o-secure], never_cache: true, record_outcome: true}
    - when: {region: EU}
      then: {only: [eu-endpoint]}

DECLARED, NEVER INFERRED — THIS IS THE WHOLE POSTURE
-----------------------------------------------------
`data_class` and `region` are supplied by the application. They are never
guessed from the prompt, because guessing means READING the prompt, and the
single property that lets this run inside a bank is that we never do.

The honest consequence, which belongs in front of a compliance buyer rather
than discovered by one: enforcement is on the DECLARED class. If an application
labels a PHI case as `general`, this faithfully enforces the wrong rule. Every
record says "the declared rule was enforced on the declared class" — never "we
verified the content".

FAIL-CLOSED, AND ONLY HERE
---------------------------
Everything else in this codebase fails open: if the meter breaks, traffic
flows. That is right for an optimizer and wrong for a compliance control — a
model allowlist that fails open sends PHI to an unapproved model, which is
precisely the harm the rule exists to prevent.

The inversion is kept as narrow as it can be:

  * if the ruleset RESOLVES, its verdict is enforced strictly — a model outside
    `only` is refused, and the call never happens;
  * if resolution itself fails, policy contributes nothing, which is the
    documented behaviour of the whole rule engine (S4-2) and leaves any limits
    declared in code untouched.

So a bug in policy evaluation degrades to "no policy", never to "deny
everything" — an engine that could take production down when IT breaks is not a
safety feature.

DENY BEATS ALLOW
----------------
When both apply, deny wins. That matches the most-restrictive combination the
rule engine already uses, and it means a broad allowlist can never quietly
re-permit something a specific rule forbade.
"""
from __future__ import annotations

import contextlib
import contextvars
import re
from typing import Any, Dict, FrozenSet, Iterator, Optional, Tuple

__all__ = [
    "STARTER_POLICIES", "starter_policy", "list_starter_policies",
    "policy_report",
    "data_class", "set_data_class", "current_data_class",
    "region", "set_region", "current_region",
    "PolicyViolation", "ModelNotPermitted", "EndpointNotPermitted",
    "ComplianceDecision", "resolve_compliance", "check_model",
    "check_endpoint",
]

# Same grammar as every other declared label in this codebase: an operator's
# identifier, never content.
_LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-]{0,63}$")


def _validate(label: Optional[str], what: str) -> Optional[str]:
    if label is None:
        return None
    label = str(label)
    if not _LABEL_RE.match(label):
        raise ValueError(
            f"{what} must be 1-64 chars of [A-Za-z0-9._-] starting "
            f"alphanumeric — a declared label such as 'PHI' or 'EU', never "
            f"content; got {label!r}")
    return label


# ── exceptions ──────────────────────────────────────────────────────────

class PolicyViolation(RuntimeError):
    """A call was refused by a compliance rule.

    Deliberately NOT a TaskLimitExceeded: a budget ceiling and a data-handling
    rule fail for different reasons, are owned by different people, and are
    handled differently by a caller. Collapsing them would make "we ran out of
    budget" indistinguishable from "that model is not approved for PHI".
    """

    def __init__(self, message: str, *, rule: Optional[str] = None,
                 data_class: Optional[str] = None,
                 region: Optional[str] = None) -> None:
        super().__init__(message)
        self.rule = rule
        self.data_class = data_class
        self.region = region


class EndpointNotPermitted(PolicyViolation):
    """The call would be processed somewhere policy does not allow.

    Separate from ModelNotPermitted because they are different failures with
    different fixes: one means "use a different model", the other means "route
    this to a different place". A residency breach and an approval breach must
    not be indistinguishable in an incident review.
    """

    def __init__(self, message: str, *, endpoint: str, permitted=None,
                 denied=None, **kw) -> None:
        super().__init__(message, **kw)
        self.endpoint = endpoint
        self.permitted = tuple(permitted or ())
        self.denied = tuple(denied or ())


class ModelNotPermitted(PolicyViolation):
    """The bound model is not permitted for this request's declared context."""

    def __init__(self, message: str, *, model: str, permitted=None,
                 denied=None, **kw) -> None:
        super().__init__(message, **kw)
        self.model = model
        self.permitted = tuple(permitted or ())
        self.denied = tuple(denied or ())


# ── declared context ────────────────────────────────────────────────────

_DATA_CLASS: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "tokeymeter_data_class", default=None)
_REGION: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "tokeymeter_region", default=None)


def set_data_class(label: Optional[str]) -> Optional[str]:
    """Bind the declared data class directly; returns the previous value."""
    prev = _DATA_CLASS.get()
    _DATA_CLASS.set(_validate(label, "data_class"))
    return prev


def current_data_class() -> Optional[str]:
    return _DATA_CLASS.get()


@contextlib.contextmanager
def data_class(label: Optional[str]) -> Iterator[None]:
    """Declare the data class for a scope — 'PHI', 'PII', 'general'.

    The application knows what it is handling; this module never guesses.
    Restores the previous binding on exit, including on exception.
    """
    token = _DATA_CLASS.set(_validate(label, "data_class"))
    try:
        yield
    finally:
        _DATA_CLASS.reset(token)


def set_region(label: Optional[str]) -> Optional[str]:
    prev = _REGION.get()
    _REGION.set(_validate(label, "region"))
    return prev


def current_region() -> Optional[str]:
    return _REGION.get()


@contextlib.contextmanager
def region(label: Optional[str]) -> Iterator[None]:
    """Declare the region a request originates from — 'EU', 'US'."""
    token = _REGION.set(_validate(label, "region"))
    try:
        yield
    finally:
        _REGION.reset(token)


# ── the resolved decision ───────────────────────────────────────────────

class ComplianceDecision:
    """What policy permits for one context. Immutable and cheap to consult.

    Resolution walks the ruleset; the per-call check is a set membership test,
    so a policy costs nothing on the hot path no matter how many rules it has.
    """

    __slots__ = ("only", "deny", "only_endpoints", "deny_endpoints",
                 "never_cache", "record_outcome", "rules", "context")

    def __init__(self, only: FrozenSet[str], deny: FrozenSet[str],
                 never_cache: bool, record_outcome: bool,
                 rules: Tuple[str, ...], context: Dict[str, Optional[str]],
                 only_endpoints: FrozenSet[str] = frozenset(),
                 deny_endpoints: FrozenSet[str] = frozenset()) -> None:
        self.only = only
        self.deny = deny
        self.only_endpoints = only_endpoints
        self.deny_endpoints = deny_endpoints
        self.never_cache = never_cache
        self.record_outcome = record_outcome
        self.rules = rules
        self.context = context

    @property
    def restricts_models(self) -> bool:
        return bool(self.only or self.deny or self.only_endpoints
                    or self.deny_endpoints)

    @property
    def restricts_endpoints(self) -> bool:
        return bool(self.only_endpoints or self.deny_endpoints)

    @property
    def is_empty(self) -> bool:
        return not (self.only or self.deny or self.only_endpoints
                    or self.deny_endpoints or self.never_cache
                    or self.record_outcome)

    def permits(self, model: Optional[str]) -> bool:
        """Deny beats allow, so a broad allowlist can never quietly re-permit
        something a specific rule forbade."""
        return _permitted(model, self.only, self.deny)

    def permits_endpoint(self, endpoint: Optional[str]) -> bool:
        """Whether the call may be PROCESSED there.

        An UNDECLARED endpoint under an endpoint allowlist is refused, unlike
        an undeclared model: residency is a claim about where processing
        happens, and "we do not know where this went" is not evidence that it
        stayed in the EU. Failing open here would make the control decorative
        for exactly the deployments that need it.
        """
        if self.only_endpoints and endpoint is None:
            return False
        return _permitted(endpoint, self.only_endpoints, self.deny_endpoints)

    def as_dict(self) -> Dict[str, Any]:
        return {"only": sorted(self.only), "deny": sorted(self.deny),
                "only_endpoints": sorted(self.only_endpoints),
                "deny_endpoints": sorted(self.deny_endpoints),
                "never_cache": self.never_cache,
                "record_outcome": self.record_outcome,
                "rules": list(self.rules),
                "context": dict(self.context)}


def _permitted(value, allow: FrozenSet[str], deny: FrozenSet[str]) -> bool:
    if value is None:
        return True
    if value in deny:
        return False
    if allow:
        return value in allow
    return True


EMPTY_DECISION = ComplianceDecision(frozenset(), frozenset(), False, False,
                                    (), {})


# Resolution is memoised per (ruleset version, context). A ruleset swap changes
# the version, so a stale decision can never outlive the policy that produced
# it. Bounded, because the context tuple is drawn from a small set of declared
# labels rather than anything user-supplied at request scale.
_MEMO: Dict[Tuple, ComplianceDecision] = {}
_MEMO_CAP = 4096


def _as_set(value) -> FrozenSet[str]:
    if value is None:
        return frozenset()
    if isinstance(value, str):
        return frozenset([value])
    try:
        return frozenset(str(v) for v in value)
    except TypeError:
        return frozenset()


def resolve_compliance(ruleset=None) -> ComplianceDecision:
    """The compliance verdict for the current declared context.

    Never raises. If the ruleset cannot be resolved, returns the empty decision
    — policy contributes nothing, which is the engine's documented fail-open
    behaviour and leaves code-declared limits untouched.
    """
    try:
        from tokeymeter.engines.governance import rules as _rules
        from tokeymeter.engines.execution import task as _task
        rs = ruleset if ruleset is not None else _rules.get_rules()
        if not getattr(rs, "rules", ()):
            return EMPTY_DECISION
        # Fast path: most policies contain no compliance actions at all, and
        # those users must not pay for a feature they do not use. Answering
        # "does this ruleset restrict anything?" once per ruleset is far
        # cheaper than assembling the declared context on every call.
        if not _has_compliance_actions(rs):
            return EMPTY_DECISION
        ctx = {
            "data_class": _DATA_CLASS.get(),
            "region": _REGION.get(),
            "agent": _task.current_agent(),
            "env": _rules.current_env(),
            "task_id": _task.current_task_id(),
            "principal": _principal_safe(),
            # Included so a rule can CONDITION on it; never read back as the
            # subject of the endpoint check, which resolves fresh per call.
            "endpoint": _endpoint_safe(),
        }
        # Key on only the fields this policy actually conditions on. Including
        # task_id unconditionally made every task a memo miss — task ids are
        # unique by construction — so the cache thrashed against its cap and
        # held one entry per task while helping nothing. Most compliance rules
        # key on data_class/region/agent, which take a handful of values.
        fields = _conditioned_fields(rs)
        key = (id(rs), rs.version) + tuple(ctx[f] for f in fields)
        hit = _MEMO.get(key)
        if hit is not None:
            return hit

        only: FrozenSet[str] = frozenset()
        deny: FrozenSet[str] = frozenset()
        only_eps: FrozenSet[str] = frozenset()
        deny_eps: FrozenSet[str] = frozenset()
        never_cache = False
        record_outcome = False
        matched = []
        for rule in rs.rules:
            try:
                if not rule.matches(ctx):
                    continue
            except Exception:
                continue
            then = rule.then
            if not any(k in then for k in _COMPLIANCE_ACTIONS):
                continue
            matched.append(rule.name or "<unnamed>")
            if "only" in then:
                # Successive allowlists INTERSECT: two rules that each permit a
                # set mean "permitted by both", never "permitted by either".
                new = _as_set(then["only"])
                only = new if not only else (only & new)
            if "deny" in then:
                deny = deny | _as_set(then["deny"])
            if "only_endpoints" in then:
                new = _as_set(then["only_endpoints"])
                only_eps = new if not only_eps else (only_eps & new)
            if "deny_endpoints" in then:
                deny_eps = deny_eps | _as_set(then["deny_endpoints"])
            if then.get("never_cache"):
                never_cache = True
            if then.get("record_outcome"):
                record_outcome = True

        decision = ComplianceDecision(only, deny, never_cache, record_outcome,
                                      tuple(matched), ctx,
                                      only_endpoints=only_eps,
                                      deny_endpoints=deny_eps)
        if len(_MEMO) >= _MEMO_CAP:
            _MEMO.clear()
        _MEMO[key] = decision
        return decision
    except Exception:
        return EMPTY_DECISION


_HAS_ACTIONS: Dict[Tuple[int, int], bool] = {}
_COMPLIANCE_ACTIONS = ("only", "deny", "only_endpoints", "deny_endpoints",
                       "never_cache", "record_outcome")


_FIELDS: Dict[Tuple[int, int], Tuple[str, ...]] = {}


def _conditioned_fields(rs) -> Tuple[str, ...]:
    """Which condition fields this ruleset actually tests, in stable order.

    Memoised per (identity, version) alongside the action check, so a swapped
    or edited policy is never answered from a stale answer.
    """
    key = (id(rs), rs.version)
    hit = _FIELDS.get(key)
    if hit is not None:
        return hit
    used = set()
    for r in rs.rules:
        for c in r.when:
            used.add(c.field)
    fields = tuple(f for f in ("data_class", "region", "agent", "env",
                               "task_id", "principal", "endpoint")
                   if f in used)
    if len(_FIELDS) >= 512:
        _FIELDS.clear()
    _FIELDS[key] = fields
    return fields


def _has_compliance_actions(rs) -> bool:
    """Whether a ruleset restricts anything at all, memoised per ruleset.

    Keyed on identity AND version so a swapped or edited policy is never
    answered from a stale verdict.
    """
    key = (id(rs), rs.version)
    hit = _HAS_ACTIONS.get(key)
    if hit is not None:
        return hit
    found = any(a in r.then for r in rs.rules for a in _COMPLIANCE_ACTIONS)
    if len(_HAS_ACTIONS) >= 512:
        _HAS_ACTIONS.clear()
    _HAS_ACTIONS[key] = found
    return found


def _endpoint_safe() -> Optional[str]:
    try:
        from tokeymeter.engines.execution.endpoint import get_endpoint
        return get_endpoint()
    except Exception:
        return None


def _principal_safe() -> Optional[str]:
    try:
        from tokeymeter.engines.governance.identity import get_principal
        return get_principal()
    except Exception:
        return None


def clear_cache() -> None:
    """Drop memoised decisions. Called when a ruleset is installed."""
    _MEMO.clear()
    _HAS_ACTIONS.clear()
    _FIELDS.clear()


# ── the per-call check ──────────────────────────────────────────────────

def check_endpoint(endpoint: Optional[str] = None,
                   decision: Optional[ComplianceDecision] = None
                   ) -> ComplianceDecision:
    """Refuse the call if policy does not allow processing where it would go.

    Residency is enforced on the DECLARED endpoint, exactly as data class is
    enforced on the declared class: the operator declares which endpoint
    identity means "in the EU", and this enforces that declaration. It is not
    a geolocation check and never claims to be.
    """
    d = decision if decision is not None else resolve_compliance()
    if not d.restricts_endpoints:
        return d
    # Read the endpoint FRESH. The decision — what policy allows — is memoised
    # and shared across calls; the subject — where THIS call is going — is not,
    # and taking it from the cached context returned a previous call's
    # destination. The model is passed in for exactly this reason; the endpoint
    # has to be too.
    ep = endpoint if endpoint is not None else _endpoint_safe()
    if d.permits_endpoint(ep):
        return d
    if ep is None:
        detail = ("no endpoint is declared, and an undeclared destination "
                  "cannot be evidence that the data stayed in region")
    elif ep in d.deny_endpoints:
        detail = "it is explicitly denied"
    else:
        detail = f"permitted endpoints are {', '.join(sorted(d.only_endpoints))}"
    where = []
    if d.context.get("data_class"):
        where.append(f"data_class={d.context['data_class']}")
    if d.context.get("region"):
        where.append(f"region={d.context['region']}")
    scope = f" for {', '.join(where)}" if where else ""
    raise EndpointNotPermitted(
        f"endpoint {ep!r} may not process this call{scope}: {detail} "
        f"(rule(s): {', '.join(d.rules) or 'none'})",
        endpoint=str(ep), permitted=d.only_endpoints, denied=d.deny_endpoints,
        rule=d.rules[0] if d.rules else None,
        data_class=d.context.get("data_class"),
        region=d.context.get("region"))


def check_model(model: Optional[str],
                decision: Optional[ComplianceDecision] = None
                ) -> ComplianceDecision:
    """Refuse the call if policy does not permit this model.

    Raises ModelNotPermitted — the ONE place in this codebase that fails
    closed, and only when the ruleset resolved successfully. Returns the
    decision so a caller can act on `never_cache` without resolving twice.
    """
    d = decision if decision is not None else resolve_compliance()
    if d.is_empty:
        return d
    check_endpoint(None, d)    # WHERE it runs, before WHICH model
    if d.permits(model):
        return d
    where = []
    if d.context.get("data_class"):
        where.append(f"data_class={d.context['data_class']}")
    if d.context.get("region"):
        where.append(f"region={d.context['region']}")
    scope = f" for {', '.join(where)}" if where else ""
    if model in d.deny:
        detail = "it is explicitly denied"
    else:
        detail = f"permitted models are {', '.join(sorted(d.only))}"
    raise ModelNotPermitted(
        f"model {model!r} is not permitted{scope}: {detail} "
        f"(rule(s): {', '.join(d.rules) or 'none'})",
        model=str(model), permitted=d.only, denied=d.deny,
        rule=d.rules[0] if d.rules else None,
        data_class=d.context.get("data_class"),
        region=d.context.get("region"))


# ── starter library ─────────────────────────────────────────────────────
#
# An empty policy file is a dead install: an officer who opens one has no way
# to know what this can express. Kyverno ships a policy library for exactly
# this reason, and it is the difference between a capability and something
# anyone actually turns on.
#
# These are STARTING POINTS, not certifications. We implement primitives; a
# regime is a mapping onto them, and the model names below are placeholders a
# customer replaces with the ones their own security review approved. Nobody
# should read "hipaa-phi-handling" as a claim that installing it makes them
# HIPAA compliant — it enforces the handling rule they declare, no more.

STARTER_POLICIES: Dict[str, Dict[str, Any]] = {
    "hipaa-phi-handling": {
        "description": ("PHI may only reach models the operator has approved, "
                        "is never retained, and every decision is recorded."),
        "replace": ["the model names in `only`"],
        "rules": [
            {"name": "phi-approved-models-only",
             "when": {"data_class": "PHI"},
             "then": {"only": ["REPLACE-WITH-YOUR-APPROVED-MODEL"],
                      "never_cache": True, "record_outcome": True}},
        ],
    },
    "eu-data-residency": {
        "description": ("EU requests are processed only on endpoints the "
                        "operator has declared EU-resident."),
        "replace": ["the endpoint names in `only_endpoints`"],
        "rules": [
            {"name": "eu-stays-in-eu",
             "when": {"region": "EU"},
             "then": {"only_endpoints": ["REPLACE-WITH-YOUR-EU-ENDPOINT"],
                      "record_outcome": True}},
        ],
    },
    "data-residency-multi-region": {
        "description": ("A worldwide residency skeleton: each origin region is "
                        "pinned to the endpoints declared resident there."),
        "replace": ["every endpoint name — one line per region you operate in"],
        "rules": [
            {"name": "eu-residency", "when": {"region": "EU"},
             "then": {"only_endpoints": ["REPLACE-EU-ENDPOINT"],
                      "record_outcome": True}},
            {"name": "uk-residency", "when": {"region": "UK"},
             "then": {"only_endpoints": ["REPLACE-UK-ENDPOINT"],
                      "record_outcome": True}},
            {"name": "us-residency", "when": {"region": "US"},
             "then": {"only_endpoints": ["REPLACE-US-ENDPOINT"],
                      "record_outcome": True}},
            {"name": "india-residency", "when": {"region": "IN"},
             "then": {"only_endpoints": ["REPLACE-IN-ENDPOINT"],
                      "record_outcome": True}},
            {"name": "australia-residency", "when": {"region": "AU"},
             "then": {"only_endpoints": ["REPLACE-AU-ENDPOINT"],
                      "record_outcome": True}},
        ],
    },
    "pii-minimisation": {
        "description": "PII is never retained, and its handling is recorded.",
        "replace": [],
        "rules": [
            {"name": "pii-never-retained",
             "when": {"data_class": "PII"},
             "then": {"never_cache": True, "record_outcome": True}},
        ],
    },
    "runaway-agent-guard": {
        "description": ("Bound every agent task: a spend ceiling, a loop "
                        "detector, and a stall detector."),
        "replace": ["the envelope, to match a task you consider expensive"],
        "rules": [
            {"name": "agent-ceiling",
             "then": {"envelope": 2.00, "reserve": 0.10, "max_repeats": 4,
                      "stall_window": 8, "enforce": True}},
        ],
    },
    "dev-and-ci-caps": {
        "description": "Development and CI run cheap and bounded.",
        "replace": ["the model name in `only`, if you pin one"],
        "rules": [
            {"name": "non-production-is-cheap",
             "when": {"env": ["dev", "ci"]},
             "then": {"envelope": 0.25, "enforce": True}},
        ],
    },
}


def list_starter_policies() -> Dict[str, str]:
    """Available starting points, name -> what it enforces."""
    return {k: v["description"] for k, v in STARTER_POLICIES.items()}


def starter_policy(name: str) -> Dict[str, Any]:
    """A starting policy as a plain mapping, ready to write to a file.

    Returned as DATA, never installed for you: a compliance rule the operator
    did not read and commit is not a control they can defend.
    """
    if name not in STARTER_POLICIES:
        raise KeyError(
            f"unknown starter policy {name!r}; available: "
            f"{', '.join(sorted(STARTER_POLICIES))}")
    entry = STARTER_POLICIES[name]
    return {"version": 1, "rules": [dict(r) for r in entry["rules"]]}


# ── policy report ───────────────────────────────────────────────────────

def policy_report(records=None) -> Dict[str, Any]:
    """What compliance policy actually did, from the ledger.

    Kyverno's PolicyReport idea, applied to AI execution: enforcement is worth
    little to an auditor without a durable, queryable record of what was
    evaluated and under what rule. Reads the same ledger everything else does,
    so this can never disagree with the chargeback or the close packet.

    Refused calls are counted separately from permitted ones because they are
    different evidence: one shows the control working, the other shows the
    volume it governs.
    """
    if records is None:
        from tokeymeter.engines.economics import savings as _sv
        records = list(_sv._tracker._iter_records())

    by_class: Dict[str, Dict[str, Any]] = {}
    by_region: Dict[str, Dict[str, Any]] = {}
    by_rule: Dict[str, int] = {}
    governed = 0
    ungoverned = 0
    models_seen: Dict[str, set] = {}

    for rec in records:
        dc = rec.get("data_class")
        rg = rec.get("region")
        rules = rec.get("policy_rules")
        model = rec.get("model")
        if not (dc or rg or rules):
            ungoverned += 1
            continue
        governed += 1
        if rules:
            for r in str(rules).split(","):
                if r:
                    by_rule[r] = by_rule.get(r, 0) + 1
        if dc:
            e = by_class.setdefault(str(dc), {"calls": 0, "models": set()})
            e["calls"] += 1
            if model:
                e["models"].add(str(model))
        if rg:
            e = by_region.setdefault(str(rg), {"calls": 0, "models": set(),
                                               "endpoints": set()})
            e["calls"] += 1
            if model:
                e["models"].add(str(model))
            ep = rec.get("endpoint_identity")
            if ep:
                e["endpoints"].add(str(ep))

    for table in (by_class, by_region):
        for e in table.values():
            e["models"] = sorted(e["models"])
            if "endpoints" in e:
                e["endpoints"] = sorted(e["endpoints"])

    return {
        "report": "policy",
        "governed_calls": governed,
        "ungoverned_calls": ungoverned,
        "by_data_class": dict(sorted(by_class.items())),
        "by_region": dict(sorted(by_region.items())),
        "rule_applications": dict(sorted(by_rule.items())),
        "note": (
            "Enforcement is on the DECLARED class. This reports that the "
            "declared rule was applied to the declared class — never that the "
            "content was inspected and classified, which this node does not "
            "do. `by_data_class.models` is the exhaustive list of models a "
            "class actually reached, and `by_region.endpoints` the exhaustive "
            "list of places each origin region was PROCESSED — the answer to "
            "'prove EU data never left the EU'. Residency is enforced on the "
            "DECLARED endpoint: the operator declares which endpoint identity "
            "is resident where, and this enforces that declaration. It is not "
            "a geolocation check and never claims to be."),
    }
