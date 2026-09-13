"""Compliance vocabulary (S4-7) — what a call is ALLOWED to do.

Until now every rule answered "how much". None answered "what is permitted", so
a compliance officer could approve this product in an afternoon and then author
nothing in it.

Pinned here, because each was a deliberate decision:

  DECLARED, NEVER INFERRED. `data_class` and `region` come from the
    application. Guessing them means reading the prompt, which is the one thing
    that lets this run inside a bank.
  FAIL-CLOSED, AND ONLY HERE. An allowlist that fails open sends PHI to an
    unapproved model — the exact harm the rule exists to prevent. But the
    inversion is narrow: a resolved ruleset is enforced strictly, while a
    ruleset that fails to resolve contributes nothing, so a bug in policy
    evaluation degrades to "no policy" and never to "deny everything".
  DENY BEATS ALLOW, and successive allowlists INTERSECT — a broad rule can
    never quietly re-permit what a specific one forbade.
  never_cache MEANS NO REUSE, not merely no persistence. Single-flight retains
    nothing and still hands caller B an answer generated for caller A.
  COMPLIANCE IS EVIDENCE, NOT JUST ENFORCEMENT. "The rule was applied" is worth
    nothing to an auditor without "to what, and under which rule".
"""
import json

import pytest

import tokeymeter
from tokeymeter.storage import MemoryStore
from tokeymeter.engines.economics.usage import set_reported_usage
from tokeymeter.engines.governance import rules as R
from tokeymeter.engines.governance import compliance as C
import tokeymeter.engines.economics.savings as sv


@pytest.fixture(autouse=True)
def _clean():
    tokeymeter.set_default_store(MemoryStore())
    tokeymeter.set_in_memory_savings(True)
    tokeymeter.reset_savings()
    tokeymeter.clear_registered_pricing()
    for m in ("approved", "unapproved", "eu-endpoint", "m"):
        tokeymeter.register_pricing(m, input_per_1m=2.5, output_per_1m=10.0)
    R.clear_rules()
    C.clear_cache()
    yield
    R.clear_rules()
    C.clear_cache()
    tokeymeter.set_in_memory_savings(False)
    tokeymeter.reset_savings()
    tokeymeter.clear_registered_pricing()


def _records():
    return list(sv._tracker._iter_records())


def _call(model, marker=None):
    @tokeymeter.cache(model=model)
    def fn(p):
        set_reported_usage(3000, 800)
        return marker or f"answer from {model}"
    return fn


# ── declared context ────────────────────────────────────────────────────

def test_data_class_and_region_bind_and_restore():
    assert C.current_data_class() is None
    with tokeymeter.data_class("PHI"), tokeymeter.region("EU"):
        assert C.current_data_class() == "PHI"
        assert C.current_region() == "EU"
    assert C.current_data_class() is None and C.current_region() is None


def test_bindings_restore_on_exception():
    with pytest.raises(ValueError):
        with tokeymeter.data_class("PHI"):
            raise ValueError("boom")
    assert C.current_data_class() is None


@pytest.mark.parametrize("bad", ["", "has space", "a/b", "x" * 65,
                                 "patient John Smith has diabetes"])
def test_labels_must_be_identifiers_not_content(bad):
    """A label carrying content would put content in the ledger."""
    with pytest.raises(ValueError):
        with tokeymeter.data_class(bad):
            pass


def test_context_reaches_the_ledger():
    """Enforcement without a record is worth nothing to an auditor."""
    R.set_rules(R.load_rules([{"name": "phi", "when": {"data_class": "PHI"},
                               "then": {"only": ["approved"]}}]))
    fn = _call("approved")
    with tokeymeter.data_class("PHI"), tokeymeter.region("EU"):
        fn("x")
    r = _records()[0]
    assert r["data_class"] == "PHI"
    assert r["region"] == "EU"
    assert r["policy_rules"] == "phi"


def test_the_record_still_carries_no_content():
    R.set_rules(R.load_rules([{"name": "phi", "when": {"data_class": "PHI"},
                               "then": {"only": ["approved"]}}]))
    fn = _call("approved", marker="the confidential clinical narrative")
    with tokeymeter.data_class("PHI"):
        fn("patient SSN 123-45-6789")
    blob = json.dumps(_records()[0])
    assert "123-45-6789" not in blob and "confidential" not in blob


# ── model restriction ───────────────────────────────────────────────────

def _phi_policy():
    return R.load_rules([{"name": "phi-only",
                          "when": {"data_class": "PHI"},
                          "then": {"only": ["approved"]}}])


def test_an_unapproved_model_is_refused():
    R.set_rules(_phi_policy())
    calls = {"n": 0}

    @tokeymeter.cache(model="unapproved")
    def fn(p):
        calls["n"] += 1
        set_reported_usage(3000, 800)
        return "x"

    with pytest.raises(tokeymeter.ModelNotPermitted) as ei:
        with tokeymeter.data_class("PHI"):
            fn("patient record")
    assert calls["n"] == 0                      # the call never happened
    assert ei.value.model == "unapproved"
    assert "approved" in str(ei.value)
    assert ei.value.data_class == "PHI"


def test_an_approved_model_is_allowed():
    R.set_rules(_phi_policy())
    fn = _call("approved")
    with tokeymeter.data_class("PHI"):
        fn("patient record")
    assert len(_records()) == 1


def test_the_rule_does_not_leak_to_undeclared_context():
    """A PHI rule must not quietly govern general traffic."""
    R.set_rules(_phi_policy())
    fn = _call("unapproved")
    fn("routine question")                      # no data_class declared
    assert len(_records()) == 1


def test_deny_beats_allow():
    R.set_rules(R.load_rules([
        {"name": "broad", "then": {"only": ["approved", "unapproved"]}},
        {"name": "specific", "then": {"deny": ["unapproved"]}},
    ]))
    fn = _call("unapproved")
    with pytest.raises(tokeymeter.ModelNotPermitted):
        fn("x")


def test_successive_allowlists_intersect():
    """Two rules that each permit a set mean 'permitted by both', never
    'permitted by either' — or a broad rule re-permits what a specific one
    forbade."""
    R.set_rules(R.load_rules([
        {"name": "phi", "when": {"data_class": "PHI"},
         "then": {"only": ["approved", "m"]}},
        {"name": "eu", "when": {"region": "EU"}, "then": {"only": ["m"]}},
    ]))
    with tokeymeter.data_class("PHI"), tokeymeter.region("EU"):
        d = C.resolve_compliance()
        assert d.permits("m") is True
        assert d.permits("approved") is False


def test_a_policy_violation_is_not_a_task_limit():
    """A budget ceiling and a data-handling rule fail for different reasons and
    are handled by different people. Collapsing them would make 'we ran out of
    budget' indistinguishable from 'that model is not approved for PHI'."""
    assert not issubclass(tokeymeter.ModelNotPermitted,
                          tokeymeter.TaskLimitExceeded)
    assert issubclass(tokeymeter.ModelNotPermitted, tokeymeter.PolicyViolation)


# ── never_cache ─────────────────────────────────────────────────────────

def test_never_cache_prevents_all_reuse():
    """Not merely persistence: single-flight retains nothing and still hands
    caller B an answer generated for caller A."""
    R.set_rules(R.load_rules([{"name": "phi", "when": {"data_class": "PHI"},
                               "then": {"never_cache": True}}]))
    calls = {"n": 0}

    @tokeymeter.cache(model="approved")
    def fn(p):
        calls["n"] += 1
        set_reported_usage(3000, 800)
        return "clinical"

    with tokeymeter.data_class("PHI"):
        for _ in range(5):
            fn("identical request")
    assert calls["n"] == 5
    assert sum(1 for r in _records() if r["hit"]) == 0


def test_never_cache_does_not_disturb_ordinary_caching():
    R.set_rules(R.load_rules([{"name": "phi", "when": {"data_class": "PHI"},
                               "then": {"never_cache": True}}]))
    calls = {"n": 0}

    @tokeymeter.cache(model="m")
    def fn(p):
        calls["n"] += 1
        set_reported_usage(3000, 800)
        return "r"

    for _ in range(5):
        fn("identical request")                 # no data_class declared
    assert calls["n"] == 1
    assert sum(1 for r in _records() if r["hit"]) == 4


# ── fail-closed, narrowly ───────────────────────────────────────────────

def test_a_broken_ruleset_contributes_nothing_rather_than_denying_everything(
        monkeypatch):
    """A bug in policy evaluation must degrade to "no policy", never to "deny
    everything" — an engine that can take production down when IT breaks is not
    a safety feature."""
    R.set_rules(_phi_policy())

    def explode(*_a, **_k):
        raise RuntimeError("policy engine broke")

    monkeypatch.setattr(C, "resolve_compliance", explode)
    fn = _call("unapproved")
    with tokeymeter.data_class("PHI"):
        fn("x")                                  # must NOT raise
    assert len(_records()) == 1


def test_no_policy_means_no_restriction():
    R.clear_rules()
    fn = _call("unapproved")
    with tokeymeter.data_class("PHI"):
        fn("x")
    assert len(_records()) == 1


def test_installing_a_ruleset_drops_stale_decisions():
    """A decision memoised against the OLD policy must never outlive it — a
    stale allowlist is a silently unenforced rule."""
    R.set_rules(_phi_policy())
    with tokeymeter.data_class("PHI"):
        assert C.resolve_compliance().permits("unapproved") is False
    R.set_rules(R.load_rules([{"name": "open", "then": {"only": ["unapproved"]}}]))
    with tokeymeter.data_class("PHI"):
        assert C.resolve_compliance().permits("unapproved") is True


# ── starter library ─────────────────────────────────────────────────────

def test_starter_policies_are_loadable():
    """An empty policy file is a dead install."""
    for name in tokeymeter.list_starter_policies():
        rs = R.load_rules(tokeymeter.starter_policy(name))
        assert len(rs) >= 1


def test_starter_policies_are_returned_as_data_not_installed():
    """A compliance rule the operator did not read and commit is not a control
    they can defend."""
    before = R.get_rules()
    tokeymeter.starter_policy("hipaa-phi-handling")
    assert R.get_rules() is before


def test_starter_policy_placeholders_are_obvious():
    """A placeholder that reads like a real model name would ship unedited."""
    pol = tokeymeter.starter_policy("hipaa-phi-handling")
    assert any("REPLACE" in m
               for r in pol["rules"] for m in r["then"].get("only", ()))


def test_unknown_starter_policy_lists_what_exists():
    with pytest.raises(KeyError) as ei:
        tokeymeter.starter_policy("gdpr-magic")
    assert "hipaa-phi-handling" in str(ei.value)


# ── the policy report ───────────────────────────────────────────────────

def test_policy_report_answers_the_auditors_question():
    """'Prove PHI never reached an unapproved model' — the exhaustive list of
    models a class actually reached IS the answer."""
    R.set_rules(_phi_policy())
    approved, unapproved = _call("approved"), _call("unapproved")
    for i in range(10):
        with tokeymeter.data_class("PHI"):
            approved(f"patient {i}")
    refused = 0
    for i in range(3):
        try:
            with tokeymeter.data_class("PHI"):
                unapproved(f"patient {i}")
        except tokeymeter.ModelNotPermitted:
            refused += 1
    for i in range(5):
        unapproved(f"routine {i}")

    rep = tokeymeter.policy_report()
    assert refused == 3
    assert rep["by_data_class"]["PHI"]["models"] == ["approved"]
    assert rep["by_data_class"]["PHI"]["calls"] == 10
    assert rep["governed_calls"] == 10
    assert rep["ungoverned_calls"] == 5
    assert rep["rule_applications"]["phi-only"] == 10


def test_policy_report_states_its_own_limit():
    """Enforcement is on the DECLARED class. Saying so is the difference
    between evidence and an overclaim."""
    note = tokeymeter.policy_report([])["note"].lower()
    assert "declared" in note and "never that the content was inspected" in note


def test_policy_report_on_an_empty_ledger():
    rep = tokeymeter.policy_report([])
    assert rep["governed_calls"] == 0 and rep["by_data_class"] == {}


def test_policy_report_is_a_pure_query():
    R.set_rules(_phi_policy())
    fn = _call("approved")
    with tokeymeter.data_class("PHI"):
        fn("x")
    before = len(_records())
    tokeymeter.policy_report()
    assert len(_records()) == before


# ── validation of the new actions ───────────────────────────────────────

@pytest.mark.parametrize("bad", [{"only": []}, {"only": 5}, {"deny": [""]},
                                 {"only": [123]}, {"never_cache": "yes"}])
def test_bad_compliance_actions_rejected_at_load(bad):
    with pytest.raises(R.RulePolicyError):
        R.load_rules([{"name": "bad", "then": bad}])


def test_a_single_model_name_is_accepted_as_a_string():
    rs = R.load_rules([{"name": "one", "then": {"only": "approved"}}])
    assert rs.rules[0].then["only"] == ("approved",)


# ── every execution route must enforce, not just the one a team happens to use ──

def test_the_kernel_path_enforces_compliance():
    """The kernel is a SECOND route into the same providers. A control that
    exists on only one path is only as strong as the path a team picks —
    verified broken before this test existed: PHI reached an unapproved model
    through Runtime.execute."""
    from tokeymeter.runtime.facade import Runtime
    R.set_rules(_phi_policy())
    calls = {"n": 0}

    def upstream(prompt, **kw):
        calls["n"] += 1
        return "clinical answer"

    rt = Runtime(call=upstream, model="unapproved")
    with pytest.raises(tokeymeter.ModelNotPermitted):
        with tokeymeter.data_class("PHI"):
            rt.execute("patient record")
    assert calls["n"] == 0


def test_the_kernel_path_allows_an_approved_model():
    from tokeymeter.runtime.facade import Runtime
    R.set_rules(_phi_policy())
    rt = Runtime(call=lambda p, **k: "ok", model="approved")
    with tokeymeter.data_class("PHI"):
        assert rt.execute("patient record") == "ok"


def test_the_async_path_enforces_compliance():
    import asyncio
    R.set_rules(_phi_policy())
    calls = {"n": 0}

    @tokeymeter.cache(model="unapproved")
    async def fn(p):
        calls["n"] += 1
        set_reported_usage(3000, 800)
        return "x"

    async def run():
        with tokeymeter.data_class("PHI"):
            await fn("patient")

    with pytest.raises(tokeymeter.ModelNotPermitted):
        asyncio.run(run())
    assert calls["n"] == 0


def test_the_streaming_path_enforces_compliance():
    import asyncio
    R.set_rules(_phi_policy())
    calls = {"n": 0}

    @tokeymeter.cache_stream(model="unapproved")
    async def fn(p):
        calls["n"] += 1
        set_reported_usage(3000, 800)
        yield "a"

    async def run():
        with tokeymeter.data_class("PHI"):
            async for _ in fn("patient"):
                pass

    with pytest.raises(tokeymeter.ModelNotPermitted):
        asyncio.run(run())
    assert calls["n"] == 0


def test_a_refusal_does_not_consume_task_budget():
    """"You may not use this model" is a different answer from "you have run
    out of money", and a caller must not be charged budget for a call policy
    never allowed to happen."""
    R.set_rules(_phi_policy())
    fn = _call("unapproved")
    with tokeymeter.task("t", max_calls=5, enforce=True) as st:
        for i in range(20):
            with pytest.raises(tokeymeter.ModelNotPermitted):
                with tokeymeter.data_class("PHI"):
                    fn(f"p{i}")
    assert st.snapshot()["calls"] == 0


def test_the_decision_memo_keys_only_on_fields_the_policy_uses():
    """Keying on task_id unconditionally made every task a memo miss — task ids
    are unique by construction — so the cache thrashed against its cap while
    helping nothing."""
    R.set_rules(_phi_policy())
    fn = _call("approved")
    for i in range(200):
        with tokeymeter.task(f"t-{i}", agent="a"), tokeymeter.data_class("PHI"):
            fn(f"{i}")
    assert len(C._MEMO) <= 4
    assert C._conditioned_fields(R.get_rules()) == ("data_class",)


def test_compliance_survives_concurrent_enforcement():
    import concurrent.futures as cf
    R.set_rules(_phi_policy())
    bad, good = _call("unapproved"), _call("approved")
    escaped = {"n": 0}
    errors = []

    def work(i):
        try:
            if i % 2:
                try:
                    with tokeymeter.data_class("PHI"):
                        bad(f"p{i}")
                    escaped["n"] += 1
                except tokeymeter.ModelNotPermitted:
                    pass
            else:
                with tokeymeter.data_class("PHI"):
                    good(f"q{i}")
        except Exception as e:                  # pragma: no cover
            errors.append(repr(e))

    with cf.ThreadPoolExecutor(16) as ex:
        list(ex.map(work, range(400)))
    assert escaped["n"] == 0 and errors == []


# ── data residency: WHERE a call is processed ───────────────────────────

def _endpoint_ctx(name):
    from tokeymeter.engines.execution.endpoint import endpoint
    return endpoint(name)


def _residency_policy():
    return R.load_rules([{"name": "eu-residency", "when": {"region": "EU"},
                          "then": {"only_endpoints": ["eu-1", "eu-2"]}}])


def test_a_model_allowlist_cannot_express_residency():
    """The gap this closes: the SAME model is deployed in many regions — that
    is the Azure, Bedrock and self-hosted shape — so restricting the model name
    lets EU data reach a US deployment of an approved model."""
    R.set_rules(R.load_rules([{"name": "eu", "when": {"region": "EU"},
                               "then": {"only": ["approved"]}}]))
    fn = _call("approved")
    with tokeymeter.region("EU"), _endpoint_ctx("us-1"):
        fn("EU citizen record")               # model allowlist alone allows it
    assert len(_records()) == 1


def test_a_permitted_endpoint_is_allowed():
    R.set_rules(_residency_policy())
    fn = _call("approved")
    with tokeymeter.region("EU"), _endpoint_ctx("eu-1"):
        fn("x")
    assert len(_records()) == 1


def test_a_foreign_endpoint_is_refused():
    R.set_rules(_residency_policy())
    calls = {"n": 0}

    @tokeymeter.cache(model="approved")
    def fn(p):
        calls["n"] += 1
        set_reported_usage(3000, 800)
        return "x"

    with pytest.raises(tokeymeter.EndpointNotPermitted) as ei:
        with tokeymeter.region("EU"), _endpoint_ctx("us-1"):
            fn("EU citizen record")
    assert calls["n"] == 0
    assert ei.value.endpoint == "us-1"
    assert ei.value.region == "EU"


def test_an_undeclared_endpoint_is_refused_under_a_residency_rule():
    """Unlike a model, an UNDECLARED destination is refused: "we do not know
    where this went" is not evidence that it stayed in the EU, and failing open
    would make the control decorative for exactly the deployments that need
    it."""
    R.set_rules(_residency_policy())
    fn = _call("approved")
    with pytest.raises(tokeymeter.EndpointNotPermitted):
        with tokeymeter.region("EU"):
            fn("x")


def test_residency_does_not_leak_to_other_regions():
    R.set_rules(_residency_policy())
    fn = _call("approved")
    with tokeymeter.region("US"), _endpoint_ctx("us-1"):
        fn("x")
    assert len(_records()) == 1


def test_the_endpoint_subject_is_read_fresh_not_from_the_memo():
    """The decision — what policy allows — is memoised and shared. The subject
    — where THIS call is going — is not. Reading it from the cached context
    returned a PREVIOUS call's destination and refused a permitted endpoint."""
    R.set_rules(_residency_policy())
    fn = _call("approved")
    with pytest.raises(tokeymeter.EndpointNotPermitted):
        with tokeymeter.region("EU"), _endpoint_ctx("us-1"):
            fn("first")
    with tokeymeter.region("EU"), _endpoint_ctx("eu-1"):
        fn("second")                          # must NOT inherit 'us-1'
    with tokeymeter.region("EU"), _endpoint_ctx("eu-2"):
        fn("third")
    assert len(_records()) == 2


def test_endpoint_denylist_beats_an_allowlist():
    R.set_rules(R.load_rules([
        {"name": "broad", "then": {"only_endpoints": ["eu-1", "eu-2"]}},
        {"name": "decommissioned", "then": {"deny_endpoints": ["eu-2"]}},
    ]))
    fn = _call("approved")
    with pytest.raises(tokeymeter.EndpointNotPermitted):
        with _endpoint_ctx("eu-2"):
            fn("x")


def test_endpoint_and_model_rules_compose():
    R.set_rules(R.load_rules([{"name": "both", "when": {"data_class": "PHI"},
                               "then": {"only": ["approved"],
                                        "only_endpoints": ["eu-1"]}}]))
    ok, bad = _call("approved"), _call("unapproved")
    with tokeymeter.data_class("PHI"), _endpoint_ctx("eu-1"):
        ok("x")
    with pytest.raises(tokeymeter.EndpointNotPermitted):
        with tokeymeter.data_class("PHI"), _endpoint_ctx("us-1"):
            ok("x")
    with pytest.raises(tokeymeter.ModelNotPermitted):
        with tokeymeter.data_class("PHI"), _endpoint_ctx("eu-1"):
            bad("x")


def test_residency_and_approval_failures_are_distinguishable():
    """A residency breach and an approval breach must not look the same in an
    incident review: one means "route it elsewhere", the other "use a different
    model"."""
    assert not issubclass(tokeymeter.EndpointNotPermitted,
                          tokeymeter.ModelNotPermitted)
    assert issubclass(tokeymeter.EndpointNotPermitted,
                      tokeymeter.PolicyViolation)


def test_a_rule_can_condition_on_the_endpoint():
    R.set_rules(R.load_rules([{"name": "legacy-cluster",
                               "when": {"endpoint": "old-1"},
                               "then": {"never_cache": True}}]))
    with _endpoint_ctx("old-1"):
        assert C.resolve_compliance().never_cache is True
    with _endpoint_ctx("new-1"):
        assert C.resolve_compliance().never_cache is False


def test_the_report_proves_where_each_region_was_processed():
    """'Prove EU data never left the EU' — the exhaustive list of endpoints a
    region was processed on IS the answer."""
    R.set_rules(_residency_policy())
    fn = _call("approved")
    for i in range(4):
        with tokeymeter.region("EU"), _endpoint_ctx("eu-1"):
            fn(f"eu-{i}")
    with pytest.raises(tokeymeter.EndpointNotPermitted):
        with tokeymeter.region("EU"), _endpoint_ctx("us-1"):
            fn("leak")
    rep = tokeymeter.policy_report()
    assert rep["by_region"]["EU"]["endpoints"] == ["eu-1"]


def test_the_multi_region_starter_pack_loads():
    pol = tokeymeter.starter_policy("data-residency-multi-region")
    assert len(R.load_rules(pol)) >= 5
    assert all("REPLACE" in ep
               for r in pol["rules"] for ep in r["then"]["only_endpoints"])


def test_the_report_states_that_residency_is_declared_not_geolocated():
    note = tokeymeter.policy_report([])["note"].lower()
    assert "not a geolocation check" in note


# ── plan must model compliance, and the ledger must support planning it ──

def test_declared_context_is_recorded_even_with_no_policy():
    """Reading these through the resolved DECISION meant they were only
    recorded when a policy already existed — so a platform owner could not plan
    a residency rule against ungoverned history, which is precisely when they
    need to. What the application declared is a fact about the call."""
    R.clear_rules()
    fn = _call("approved")
    with tokeymeter.region("EU"), tokeymeter.data_class("PHI"), _endpoint_ctx("us-1"):
        fn("x")
    r = _records()[0]
    assert r["region"] == "EU"
    assert r["data_class"] == "PHI"
    assert r["endpoint_identity"] == "us-1"
    assert r["policy_rules"] is None          # nothing applied, correctly


def test_plan_reports_calls_a_residency_rule_would_refuse():
    """ENFORCEMENT AND SIMULATION MUST MOVE TOGETHER. A plan reporting "0 would
    halt" for a rule that would refuse half the estate's EU traffic is worse
    than no plan — and a compliance rule is the one you must never push
    blind."""
    from tokeymeter.engines.governance import plan as P
    R.clear_rules()
    fn = _call("approved")
    for i in range(10):
        with tokeymeter.task(f"t-{i}", agent="svc"), tokeymeter.region("EU"), \
                _endpoint_ctx("eu-1" if i % 2 else "us-1"):
            fn(f"p{i}")
    proposed = R.load_rules([{"name": "eu", "when": {"region": "EU"},
                              "then": {"only_endpoints": ["eu-1"]}}])
    plan = P.plan_report(proposed, records=_records())
    assert plan["effect"]["compliance_refusals"] == 5
    assert plan["refusals_by_reason"]["endpoint_not_permitted"] == 5
    assert "would be REFUSED" in P.render_plan(plan)


def test_plan_reports_calls_a_model_rule_would_refuse():
    from tokeymeter.engines.governance import plan as P
    R.clear_rules()
    fn = _call("unapproved")
    for i in range(4):
        with tokeymeter.task(f"t-{i}", agent="clinical"), \
                tokeymeter.data_class("PHI"):
            fn(f"p{i}")
    proposed = R.load_rules([{"name": "phi", "when": {"data_class": "PHI"},
                              "then": {"only": ["approved"]}}])
    plan = P.plan_report(proposed, records=_records())
    assert plan["effect"]["compliance_refusals"] == 4
    assert plan["refusals_by_reason"]["model_not_permitted"] == 4


def test_plan_mirrors_the_undeclared_endpoint_asymmetry():
    """The simulator must reproduce enforcement exactly, including that an
    undeclared endpoint is refused while an undeclared model is not."""
    from tokeymeter.engines.governance import plan as P
    R.clear_rules()
    fn = _call("approved")
    for i in range(3):
        with tokeymeter.task(f"t-{i}", agent="svc"), tokeymeter.region("EU"):
            fn(f"p{i}")                        # no endpoint declared
    proposed = R.load_rules([{"name": "eu", "when": {"region": "EU"},
                              "then": {"only_endpoints": ["eu-1"]}}])
    plan = P.plan_report(proposed, records=_records())
    assert plan["refusals_by_reason"]["endpoint_undeclared"] == 3


def test_plan_reports_no_refusals_for_compliant_history():
    """The warning must not fire on clean traffic, or it becomes noise."""
    from tokeymeter.engines.governance import plan as P
    R.clear_rules()
    fn = _call("approved")
    for i in range(6):
        with tokeymeter.task(f"t-{i}", agent="svc"), tokeymeter.region("EU"), \
                _endpoint_ctx("eu-1"):
            fn(f"p{i}")
    proposed = R.load_rules([{"name": "eu", "when": {"region": "EU"},
                              "then": {"only_endpoints": ["eu-1"]}}])
    plan = P.plan_report(proposed, records=_records())
    assert plan["effect"]["compliance_refusals"] == 0
    assert "would be REFUSED" not in P.render_plan(plan)
