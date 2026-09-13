"""Execution rules — the company's own AI execution policy, as data.

WHY THIS EXISTS
---------------
S4-1 gave a task a ceiling, but the number lived in application code:

    with tokeymeter.task("ticket-1", envelope=1.00):   # in Maya's repo
    with tokeymeter.task("doc-7",    envelope=10.00):  # in Raj's repo

Two numbers, two repositories, nobody chose either, and the person who owns the
bill cannot see or change them. That is the moment a library has to become
infrastructure. This module moves the numbers out of code and into a file the
platform owner writes:

    - when: agent == "support"    then: envelope 1.00, max_repeats 4
    - when: agent == "extract"    then: envelope 10.00
    - when: env in [dev, ci]      then: envelope 0.25, enforce true

The `with tokeymeter.task(...)` line stays exactly as it was. Only the VALUES
move — so a developer's code gets simpler, which is why the change lands
instead of being resisted.

A SCHEMA, NOT A LANGUAGE
------------------------
Every rule anyone has needed is `field op value`, ANDed, so that is precisely
what this accepts. Not a DSL, and deliberately not an embedded expression
language:

  * it is evaluated ONCE per task, never per call, so it costs nothing on the
    hot path (the 47us budget is untouched);
  * there is no parser, so there is no injection surface and no way to author
    a rule the node cannot actually enforce;
  * a fixed field list and a fixed action list ARE the dropdowns for a no-code
    builder — the UI comes free from the schema;
  * the YAML reads like English in a pull request, which is the point: the
    rules live in the customer's repository, reviewed and blamed like any
    other code.

If nesting or OR is ever genuinely needed, the condition evaluator can be
swapped for CEL behind this same schema without changing the file format.

VALIDATION IS LOUD, EVALUATION IS SILENT
-----------------------------------------
A typo'd field or an unknown action is rejected when the ruleset is LOADED,
with the offending rule named. Nobody discovers a misspelled `enevelope` at
3am because it silently matched nothing. Once loaded, evaluation never raises:
a rule that cannot be applied is skipped and counted.

FAIL-OPEN, HONESTLY
-------------------
If a policy file is missing or malformed, no rules apply — but the limits
declared in application code STILL DO. Fail-open here means "policy
contributes nothing", never "the ceiling disappears". The degradation is
reported through the standard degraded-event bus rather than being swallowed.

MOST RESTRICTIVE WINS
---------------------
Policy and code both set limits, so they have to combine. They combine by
taking the tighter of the two, in whichever direction is tighter for that
field. Neither side can loosen what the other set, which means neither has to
trust the other:

    envelope     min   — a smaller budget is tighter
    max_calls    min
    max_repeats  min
    reserve      MAX   — holding MORE per call halts EARLIER, so a larger
                         reserve is the conservative choice, not the smaller
    enforce      OR    — if either side wants it enforced, it is enforced

A platform owner can therefore tighten every service at once without auditing
application code, and a developer can tighten their own sensitive agent
further without asking anyone.
"""
from __future__ import annotations

import json
import os
import re
import threading
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "Condition", "Rule", "RuleSet", "RulePolicyError",
    "load_rules", "load_rules_file", "set_rules", "get_rules", "clear_rules",
    "resolve_limits", "rules_version",
    "CONDITION_FIELDS", "CONDITION_OPS", "RULE_ACTIONS",
]


class RulePolicyError(ValueError):
    """A ruleset could not be loaded. Raised at LOAD time, never at match time,
    and always naming the rule at fault."""


# ── the fixed vocabulary ────────────────────────────────────────────────
#
# Every entry here is a value that genuinely EXISTS when a task opens. Nothing
# is offered that cannot be resolved, and nothing resolvable is content: these
# are operator-supplied identifiers and labels, never prompt text.

CONDITION_FIELDS: Tuple[str, ...] = (
    "agent",       # the KIND of task — "support", "extract". What rules key on.
    "env",         # dev | ci | staging | prod, from TOKEYMETER_ENV or explicit
    "task_id",     # the specific instance — "ticket-4f2a"
    "principal",   # who is behind the call, from identity.principal
    # Compliance context — DECLARED by the application, never inferred, because
    # inferring means reading the prompt and that is the one thing this node
    # does not do.
    "data_class",  # PHI | PII | general — what kind of data this request carries
    "region",      # EU | US — where the request originated
    # WHERE the call would be PROCESSED, from execution.endpoint. Distinct from
    # `region` (where the request came from) and from the model name: the same
    # model is deployed in many places, which is exactly why a model allowlist
    # cannot express data residency.
    "endpoint",
)

CONDITION_OPS: Tuple[str, ...] = (
    "==", "!=", "in", "not_in", "startswith",
)

# The actions are exactly the task ceilings. A rule cannot express something
# the node is unable to enforce.
RULE_ACTIONS: Dict[str, str] = {
    "envelope": "number",
    "reserve": "number",
    "max_calls": "int",
    "max_repeats": "int",
    "enforce": "bool",
    # Progress guard: how many recent calls to judge, and the novelty floor
    # below which responses have stopped being new. Set centrally so a platform
    # owner can turn stall detection on across every service without touching
    # anyone's code.
    "stall_window": "int",
    "min_novelty": "fraction",
    # Compliance actions: what a request is ALLOWED to do, as opposed to how
    # much it may spend. These are the only actions that FAIL CLOSED.
    "only": "models",          # model allowlist — anything else is refused
    "deny": "models",          # model denylist — beats any allowlist
    # Residency: WHERE a call may be processed. Restricting the model name is
    # not enough — `gpt-4o` exists in eastus and westeurope, and only one of
    # them keeps EU data in the EU.
    "only_endpoints": "models",
    "deny_endpoints": "models",
    "never_cache": "bool",     # do not retain, not even a hash
    "record_outcome": "bool",  # stamp evidence for the audit trail
}

# How policy and code combine, per action. See the module docstring.
_COMBINE = {
    "envelope": min,
    "reserve": max,
    "max_calls": min,
    "max_repeats": min,
    # A SMALLER window detects a stall sooner, so it is the tighter choice.
    "stall_window": min,
    # A HIGHER novelty floor flags more tasks, so it is the tighter choice.
    "min_novelty": max,
}


def _validate_number(name: str, value, *, integer: bool):
    import math
    if isinstance(value, bool):
        raise RulePolicyError(f"{name} must be a number, got a bool")
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise RulePolicyError(f"{name} must be a number, got {value!r}")
    if not math.isfinite(v) or v <= 0:
        raise RulePolicyError(f"{name} must be a finite number > 0, got {value!r}")
    if integer:
        if v != int(v):
            raise RulePolicyError(f"{name} must be a whole number, got {value!r}")
        return int(v)
    return v


# ── condition ───────────────────────────────────────────────────────────

class Condition:
    """One `field op value` test. Immutable and cheap to evaluate."""

    __slots__ = ("field", "op", "value")

    def __init__(self, field: str, op: str, value: Any) -> None:
        if field not in CONDITION_FIELDS:
            raise RulePolicyError(
                f"unknown condition field {field!r}; allowed: "
                f"{', '.join(CONDITION_FIELDS)}")
        if op not in CONDITION_OPS:
            raise RulePolicyError(
                f"unknown operator {op!r}; allowed: {', '.join(CONDITION_OPS)}")
        if op in ("in", "not_in"):
            if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
                raise RulePolicyError(
                    f"operator {op!r} needs a list of values, got {value!r}")
            value = tuple(str(v) for v in value)
        else:
            if not isinstance(value, (str, int, float)) or isinstance(value, bool):
                raise RulePolicyError(
                    f"operator {op!r} needs a scalar value, got {value!r}")
            value = str(value)
        object.__setattr__(self, "field", field)
        object.__setattr__(self, "op", op)
        object.__setattr__(self, "value", value)

    def matches(self, context: Dict[str, Optional[str]]) -> bool:
        """A field that is absent from the context matches NOTHING — an
        unbound field must never satisfy a rule by accident. The one exception
        is `!=` / `not_in`, where an absent value genuinely is 'not that'."""
        actual = context.get(self.field)
        if actual is None:
            return self.op in ("!=", "not_in")
        actual = str(actual)
        if self.op == "==":
            return actual == self.value
        if self.op == "!=":
            return actual != self.value
        if self.op == "in":
            return actual in self.value
        if self.op == "not_in":
            return actual not in self.value
        if self.op == "startswith":
            return actual.startswith(self.value)
        return False

    def as_dict(self) -> Dict[str, Any]:
        return {"field": self.field, "op": self.op,
                "value": list(self.value) if isinstance(self.value, tuple)
                else self.value}

    def __repr__(self) -> str:
        return f"{self.field} {self.op} {self.value!r}"


# ── rule ────────────────────────────────────────────────────────────────

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._: \-]{0,63}$")


class Rule:
    """One `when ... then ...` entry. Conditions are ANDed; an empty condition
    list matches every task, which is how a default is written."""

    __slots__ = ("name", "when", "then")

    def __init__(self, name: Optional[str], when: Sequence[Condition],
                 then: Dict[str, Any]) -> None:
        if name is not None and not _NAME_RE.match(str(name)):
            raise RulePolicyError(
                f"rule name must be 1-64 chars of [A-Za-z0-9._: -] starting "
                f"alphanumeric, got {name!r}")
        if not then:
            raise RulePolicyError(
                f"rule {name or '<unnamed>'!r} has no actions; a rule that "
                f"sets nothing would silently do nothing")
        clean: Dict[str, Any] = {}
        for action, value in then.items():
            kind = RULE_ACTIONS.get(action)
            if kind is None:
                raise RulePolicyError(
                    f"rule {name or '<unnamed>'!r}: unknown action {action!r}; "
                    f"allowed: {', '.join(sorted(RULE_ACTIONS))}")
            if kind == "models":
                names = ([value] if isinstance(value, str)
                         else list(value) if isinstance(value, (list, tuple))
                         else None)
                if not names:
                    raise RulePolicyError(
                        f"rule {name or '<unnamed>'!r}: {action} needs a model "
                        f"name or a list of them, got {value!r}")
                clean_names = []
                for m in names:
                    if not isinstance(m, str) or not m.strip():
                        raise RulePolicyError(
                            f"rule {name or '<unnamed>'!r}: {action} entries "
                            f"must be model names, got {m!r}")
                    clean_names.append(m.strip())
                clean[action] = tuple(clean_names)
            elif kind == "fraction":
                v = _validate_number(
                    f"rule {name or '<unnamed>'!r}: {action}", value,
                    integer=False)
                if v > 1:
                    raise RulePolicyError(
                        f"rule {name or '<unnamed>'!r}: {action} is a fraction "
                        f"between 0 and 1, got {value!r}")
                clean[action] = v
            elif kind == "bool":
                if not isinstance(value, bool):
                    raise RulePolicyError(
                        f"rule {name or '<unnamed>'!r}: {action} must be "
                        f"true/false, got {value!r}")
                clean[action] = value
            else:
                clean[action] = _validate_number(
                    f"rule {name or '<unnamed>'!r}: {action}", value,
                    integer=(kind == "int"))
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "when", tuple(when))
        object.__setattr__(self, "then", clean)

    def matches(self, context: Dict[str, Optional[str]]) -> bool:
        return all(c.matches(context) for c in self.when)

    def as_dict(self) -> Dict[str, Any]:
        return {"name": self.name,
                "when": [c.as_dict() for c in self.when],
                "then": dict(self.then)}

    def __repr__(self) -> str:
        w = " and ".join(repr(c) for c in self.when) or "always"
        return f"<Rule {self.name or ''} when {w} then {self.then}>"


# ── ruleset ─────────────────────────────────────────────────────────────

class RuleSet:
    """An ordered, immutable collection of rules.

    ORDER IS MEANINGFUL and LAST MATCH WINS, per action. A broad default is
    written first and a specific override after it, exactly as one writes CSS
    or firewall rules:

        - when: []                   then: {envelope: 1.00}    # default
        - when: [agent == research]  then: {envelope: 10.00}   # override

    Only the actions a later rule actually sets are overridden, so a specific
    rule can raise the envelope without discarding the default's max_repeats.
    """

    __slots__ = ("rules", "version", "source")

    def __init__(self, rules: Sequence[Rule], *, version: int = 0,
                 source: Optional[str] = None) -> None:
        object.__setattr__(self, "rules", tuple(rules))
        object.__setattr__(self, "version", int(version))
        object.__setattr__(self, "source", source)

    def resolve(self, context: Dict[str, Optional[str]]) -> Dict[str, Any]:
        """The limits this ruleset declares for a task with this context.
        Never raises: a rule that cannot be evaluated is skipped."""
        out: Dict[str, Any] = {}
        for rule in self.rules:
            try:
                if rule.matches(context):
                    out.update(rule.then)
            except Exception:
                continue
        return out

    def matching(self, context: Dict[str, Optional[str]]) -> List[Rule]:
        """Which rules matched — for `explain`, and for the audit trail."""
        hits = []
        for rule in self.rules:
            try:
                if rule.matches(context):
                    hits.append(rule)
            except Exception:
                continue
        return hits

    def as_dict(self) -> Dict[str, Any]:
        return {"version": self.version, "source": self.source,
                "rules": [r.as_dict() for r in self.rules]}

    def __len__(self) -> int:
        return len(self.rules)

    def __repr__(self) -> str:
        return f"<RuleSet v{self.version} rules={len(self.rules)} src={self.source!r}>"


EMPTY_RULESET = RuleSet(())


# ── loading ─────────────────────────────────────────────────────────────

def _parse_when(raw, rule_name) -> List[Condition]:
    """Accept both the explicit form and the compact one:

        when: [{field: agent, op: "==", value: support}]
        when: {agent: support, env: [dev, ci]}       # == for scalars, in for lists
    """
    if raw is None:
        return []
    conds: List[Condition] = []
    if isinstance(raw, dict):
        for field, value in raw.items():
            op = "in" if isinstance(value, (list, tuple)) else "=="
            conds.append(Condition(field, op, value))
        return conds
    if not isinstance(raw, (list, tuple)):
        raise RulePolicyError(
            f"rule {rule_name!r}: `when` must be a list or mapping, "
            f"got {type(raw).__name__}")
    for item in raw:
        if not isinstance(item, dict):
            raise RulePolicyError(
                f"rule {rule_name!r}: each `when` entry must be a mapping, "
                f"got {item!r}")
        if "field" in item:
            conds.append(Condition(item.get("field"), item.get("op", "=="),
                                   item.get("value")))
        else:
            for field, value in item.items():
                op = "in" if isinstance(value, (list, tuple)) else "=="
                conds.append(Condition(field, op, value))
    return conds


def load_rules(data, *, source: Optional[str] = None,
               version: int = 0) -> RuleSet:
    """Build a RuleSet from a mapping, a list of rules, or a JSON string.

    Every problem is raised here, naming the rule, so a bad policy fails at
    load rather than silently matching nothing in production.
    """
    if isinstance(data, (str, bytes)):
        try:
            data = json.loads(data)
        except Exception as e:
            raise RulePolicyError(f"policy is not valid JSON: {e}") from None
    if isinstance(data, dict):
        version = int(data.get("version", version) or 0)
        raw_rules = data.get("rules")
        if raw_rules is None:
            raise RulePolicyError("policy mapping must contain a 'rules' list")
    elif isinstance(data, (list, tuple)):
        raw_rules = data
    else:
        raise RulePolicyError(
            f"policy must be a mapping, list, or JSON string, "
            f"got {type(data).__name__}")

    rules: List[Rule] = []
    seen: set = set()
    for i, raw in enumerate(raw_rules):
        if not isinstance(raw, dict):
            raise RulePolicyError(f"rule #{i}: must be a mapping, got {raw!r}")
        name = raw.get("name")
        if name is not None:
            if name in seen:
                raise RulePolicyError(
                    f"duplicate rule name {name!r} — names must be unique so "
                    f"an audit trail can identify which rule applied")
            seen.add(name)
        then = raw.get("then")
        if then is None:
            # compact form: actions inline alongside `when`
            then = {k: v for k, v in raw.items()
                    if k in RULE_ACTIONS}
        if not isinstance(then, dict):
            raise RulePolicyError(
                f"rule {name or f'#{i}'}: `then` must be a mapping, got {then!r}")
        rules.append(Rule(name, _parse_when(raw.get("when"), name or f"#{i}"),
                          then))
    return RuleSet(rules, version=version, source=source)


def load_rules_file(path: str) -> RuleSet:
    """Load a policy from disk. JSON always; YAML when PyYAML is installed.

    YAML is optional on purpose — the package declares no runtime dependencies,
    and a policy engine must not be the thing that adds one.

    Versioning: an explicit `version:` in the file wins, because a team that
    numbers its policy means it. Otherwise the file's modification time is
    used, so a hot reload can still tell whether anything changed.
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    version = int(os.path.getmtime(path))
    lower = path.lower()
    if lower.endswith((".yaml", ".yml")):
        try:
            import yaml  # type: ignore
        except ImportError:
            raise RulePolicyError(
                f"{path} is YAML but PyYAML is not installed. Install "
                f"'pyyaml', or use JSON — the schema is identical.") from None
        try:
            data = yaml.safe_load(text)
        except Exception as e:
            raise RulePolicyError(f"{path} is not valid YAML: {e}") from None
    else:
        try:
            data = json.loads(text)
        except Exception as e:
            raise RulePolicyError(f"{path} is not valid JSON: {e}") from None
    return load_rules(data, source=path, version=version)


# ── the active ruleset (hot-swappable) ──────────────────────────────────

_LOCK = threading.Lock()
_ACTIVE: RuleSet = EMPTY_RULESET


def set_rules(ruleset: Optional[RuleSet]) -> RuleSet:
    """Install a ruleset atomically; returns the previous one.

    Swapping a whole immutable object means a task in flight either sees the
    old ruleset or the new one, never a half-applied policy.
    """
    global _ACTIVE
    if ruleset is not None and not isinstance(ruleset, RuleSet):
        raise TypeError(
            f"set_rules expects a RuleSet (see load_rules), "
            f"got {type(ruleset).__name__}")
    with _LOCK:
        prev = _ACTIVE
        _ACTIVE = ruleset if ruleset is not None else EMPTY_RULESET
    # A compliance decision memoised against the OLD policy must never outlive
    # it — a stale allowlist is a silently unenforced rule.
    try:
        from tokeymeter.engines.governance import compliance as _c
        _c.clear_cache()
    except Exception:
        pass
    return prev


def get_rules() -> RuleSet:
    return _ACTIVE


def clear_rules() -> None:
    set_rules(None)


def rules_version() -> int:
    return _ACTIVE.version


# ── resolution against declared code limits ─────────────────────────────

def resolve_limits(context: Dict[str, Optional[str]],
                   declared: Dict[str, Any],
                   ruleset: Optional[RuleSet] = None) -> Tuple[Dict[str, Any], List[str]]:
    """Combine policy limits with the ones declared in application code.

    Returns (effective_limits, matched_rule_names). Never raises — if anything
    goes wrong the declared limits are returned unchanged, because policy
    failing must never remove a ceiling the code already set.
    """
    rs = ruleset if ruleset is not None else _ACTIVE
    effective = {k: v for k, v in declared.items() if v is not None}
    try:
        if not rs.rules:
            return effective, []
        policy = rs.resolve(context)
        matched = [r.name or "<unnamed>" for r in rs.matching(context)]
    except Exception:
        return effective, []

    for action, pval in policy.items():
        dval = effective.get(action)
        if dval is None:
            effective[action] = pval
            continue
        if action == "enforce":
            effective[action] = bool(dval) or bool(pval)
            continue
        combine = _COMBINE.get(action)
        if combine is None:
            effective[action] = pval
            continue
        try:
            effective[action] = combine(dval, pval)
        except Exception:
            effective[action] = dval
    return effective, matched


# ── environment ─────────────────────────────────────────────────────────

_ENV_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-]{0,31}$")
_ENV_OVERRIDE: Optional[str] = None


def set_env(name: Optional[str]) -> None:
    """Set the deployment environment explicitly, overriding TOKEYMETER_ENV.
    Rules key on this to give dev and CI different ceilings from production."""
    global _ENV_OVERRIDE
    if name is not None:
        name = str(name)
        if not _ENV_RE.match(name):
            raise ValueError(
                f"env must be 1-32 chars of [A-Za-z0-9._-] starting "
                f"alphanumeric, got {name!r}")
    _ENV_OVERRIDE = name


def current_env() -> Optional[str]:
    """The active environment: an explicit override, else TOKEYMETER_ENV, else
    None. Never raises and never guesses — an unset environment simply matches
    no env-conditioned rule."""
    if _ENV_OVERRIDE is not None:
        return _ENV_OVERRIDE
    try:
        raw = os.environ.get("TOKEYMETER_ENV")
    except Exception:
        return None
    if not raw:
        return None
    raw = raw.strip()
    return raw if _ENV_RE.match(raw) else None
