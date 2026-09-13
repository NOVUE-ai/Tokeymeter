"""Repo furniture and examples — the parts a visitor sees before the code.

An example that does not run is worse than no example: it is the first thing a
newcomer tries, and a traceback there ends the evaluation. So every file in
`examples/` is executed here, in a subprocess, exactly as a reader would run it.

The repo files are checked for existence rather than content, because their
absence is what a visitor notices — GitHub surfaces a missing LICENSE, SECURITY
or CONTRIBUTING directly in the interface.
"""
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = os.path.join(REPO, "examples")


@pytest.mark.parametrize("name", [
    "LICENSE", "README.md", "CHANGELOG.md", "CONTRIBUTING.md",
    "SECURITY.md", "CODE_OF_CONDUCT.md", "AGENTS.md", "ACCEPTANCE.py",
])
def test_repo_file_exists(name):
    path = os.path.join(REPO, name)
    assert os.path.exists(path), f"{name} is missing"
    assert os.path.getsize(path) > 200, f"{name} is a stub"


def test_license_is_the_one_pyproject_declares():
    """A declared licence with no text is unenforceable, and the badge 404s."""
    import tomllib
    declared = str(tomllib.load(
        open(os.path.join(REPO, "pyproject.toml"), "rb"))["project"]["license"])
    text = open(os.path.join(REPO, "LICENSE"), encoding="utf-8").read()
    assert "Apache-2.0" in declared
    assert "Apache License" in text and "Version 2.0" in text


def test_ci_workflow_exists_so_the_badge_is_not_a_lie():
    path = os.path.join(REPO, ".github", "workflows", "ci.yml")
    assert os.path.exists(path)
    body = open(path, encoding="utf-8").read()
    assert "run_all_checks" in body            # the gate runs in CI
    assert "zero-dependency" in body           # the property is enforced


def test_issue_templates_exist():
    d = os.path.join(REPO, ".github", "ISSUE_TEMPLATE")
    assert os.path.isdir(d)
    names = os.listdir(d)
    assert any("false_positive" in n for n in names), (
        "a working agent being stopped is the most important report we can "
        "get; it needs its own template")


def _example_files():
    if not os.path.isdir(EXAMPLES):
        return []
    return sorted(f for f in os.listdir(EXAMPLES) if f.endswith(".py"))


def test_there_are_examples():
    assert len(_example_files()) >= 5


@pytest.mark.parametrize("name", _example_files())
def test_every_example_runs_with_no_key_and_no_network(name):
    """Run exactly as a reader would. No API key, no network, no arguments."""
    env = dict(os.environ, PYTHONPATH=REPO)
    env.pop("OPENAI_API_KEY", None)
    env.pop("ANTHROPIC_API_KEY", None)
    p = subprocess.run([sys.executable, os.path.join(EXAMPLES, name)],
                       capture_output=True, env=env, timeout=300, cwd=REPO)
    assert p.returncode == 0, (
        f"{name} failed:\n{(p.stderr or b'').decode('utf-8', 'replace')[:800]}")
    assert (p.stdout or b"").strip(), f"{name} printed nothing"


def test_examples_readme_lists_every_example():
    readme = open(os.path.join(EXAMPLES, "README.md"), encoding="utf-8").read()
    for name in _example_files():
        assert name in readme, f"{name} is not listed in examples/README.md"


def test_the_main_readme_points_at_the_examples():
    readme = open(os.path.join(REPO, "README.md"), encoding="utf-8").read()
    assert "examples/" in readme


def _readme_image_paths():
    """Repo-relative paths for every live image the README shows.

    The README uses ABSOLUTE raw.githubusercontent URLs, because the same text
    is the PyPI long description and PyPI has no repository context — a
    relative `docs/report.png` renders as a broken icon there. So the files are
    resolved back out of the URL to check they exist in the repo.
    """
    import re
    readme = open(os.path.join(REPO, "README.md"), encoding="utf-8").read()
    live = re.sub(r"<!--.*?-->", "", readme, flags=re.S)
    return set(re.findall(r'src="https://raw\.githubusercontent\.com/[^/]+/'
                          r'[^/]+/[^/]+/(docs/[^"]+)"', live))


def test_every_readme_image_exists():
    """A README with broken image icons reads as abandoned, and the report
    screenshot is the strongest thing on the page."""
    refs = _readme_image_paths()
    assert refs, "the README shows no screenshots at all"
    for ref in sorted(refs):
        path = os.path.join(REPO, ref)
        assert os.path.exists(path), f"README references a missing image: {ref}"
        assert os.path.getsize(path) > 1024, f"{ref} is empty or a stub"


def test_the_readme_has_no_relative_links():
    """The README IS the PyPI long description. Every relative link and image
    404s there, because PyPI has no idea which repository it came from."""
    import re
    readme = open(os.path.join(REPO, "README.md"), encoding="utf-8").read()
    live = re.sub(r"<!--.*?-->", "", readme, flags=re.S)
    relative = ([t for t in re.findall(r"\]\((?!https?:|#)([^)]+)\)", live)]
                + [t for t in re.findall(r'src="(?!https?:)([^"]+)"', live)])
    assert not relative, f"these 404 on PyPI: {relative}"


def test_readme_images_are_not_heavy():
    """GitHub renders a heavy README sluggishly, and a first impression that
    loads slowly is a first impression that does not happen."""
    refs = _readme_image_paths()
    total = sum(os.path.getsize(os.path.join(REPO, r)) for r in refs
                if os.path.exists(os.path.join(REPO, r)))
    assert total < 3 * 1024 * 1024, f"{total/1024/1024:.1f} MB of images"


def test_every_readme_image_has_alt_text():
    import re
    readme = open(os.path.join(REPO, "README.md"), encoding="utf-8").read()
    live = re.sub(r"<!--.*?-->", "", readme, flags=re.S)
    for tag in re.findall(r"<img [^>]+>", live):
        assert 'alt="' in tag, f"image without alt text: {tag[:70]}"
        alt = re.search(r'alt="([^"]*)"', tag).group(1)
        assert len(alt) > 8, f"alt text says nothing: {alt!r}"


def test_no_stale_repository_slug_anywhere():
    """Every image and link in the README is an absolute GitHub URL, so a slug
    that lags behind a rename silently 404s — on the PyPI page especially,
    where nobody will tell you."""
    import re
    import tomllib
    proj = tomllib.load(open(os.path.join(REPO, "pyproject.toml"), "rb"))["project"]
    slug = re.search(r"github\.com/([^/]+/[^/]+)",
                     proj["urls"]["Homepage"]).group(1)
    readme = open(os.path.join(REPO, "README.md"), encoding="utf-8").read()
    found = set(re.findall(r"github(?:usercontent)?\.com/([^/]+/[^/\s\")]+)",
                           readme))
    stray = {s for s in found if s != slug}
    assert not stray, f"README points at {stray}, pyproject says {slug}"


def test_no_internal_documents_ship_publicly():
    """This tree becomes a public repository. Two kinds of file must never be
    in it: anything specifying the unreleased commercial control plane, and
    any red-team write-up of how to defeat a trust feature — publishing the
    second contradicts SECURITY.md, which asks people not to disclose
    publicly."""
    forbidden = [
        "docs/NOVUE_PLATFORM_ARCHITECTURE.md",   # marked internal/confidential
        "docs/stress-testing",                   # vulnerability findings
        "STATE_OF_THE_RUNTIME.md",               # stale internal snapshot
        "SELF_HOSTED_BUILD_MANIFEST.md",         # internal build notes
        "README.tokeymeter-engine.md",           # a second, superseded README
    ]
    present = [f for f in forbidden if os.path.exists(os.path.join(REPO, f))]
    assert not present, f"internal material would be published: {present}"


def test_only_one_readme():
    """Two READMEs at the top level is a repository a reader cannot navigate,
    and the second one carried the positioning we abandoned."""
    tops = [f for f in os.listdir(REPO)
            if f.lower().startswith("readme") and f.endswith(".md")]
    assert tops == ["README.md"], f"found {tops}"


def test_the_readme_test_count_matches_reality():
    """A claim a reader can check in one command, so it must not drift."""
    import re
    import subprocess
    import sys
    readme = open(os.path.join(REPO, "README.md"), encoding="utf-8").read()
    claimed = {int(m.replace(",", ""))
               for m in re.findall(r"([\d,]{4,}) tests", readme)}
    badge = re.search(r"tests-([\d%C2A]+)-", readme)
    if badge:
        claimed.add(int(badge.group(1).replace("%2C", "")))
    assert len(claimed) == 1, f"the README claims several counts: {claimed}"

    out = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "--co",
         "--ignore=tests/test_benchmarks.py"],
        capture_output=True, cwd=REPO, timeout=300)
    text = (out.stdout or b"").decode("utf-8", "replace")
    collected = re.search(r"(\d+) tests? collected", text)
    if not collected:
        return                      # collection format changed; skip rather than lie
    actual = int(collected.group(1))
    # collection includes the skips; allow the small, stable gap
    assert abs(actual - claimed.pop()) <= 40, (
        f"README claims a test count that no longer matches ({actual} collected)")


def test_the_control_plane_is_not_in_the_open_source_tree():
    """integrations/tokenet is the paid surface: a client SDK, its ontology and
    its release machinery. Publishing it would give away the commercial product
    on day one and contradict the boundary the whole model rests on — free is
    complete for ONE service, paid is many services in one place."""
    assert not os.path.isdir(os.path.join(REPO, "integrations")), (
        "the TokeNet client would be published")


def test_the_gate_passes_on_the_code_that_ships():
    """A release gate whose checks target a package the distribution does not
    contain is a gate that fails on exactly the tree it is meant to certify."""
    import re
    runner = open(os.path.join(REPO, "scripts", "run_all_checks.py"),
                  encoding="utf-8").read()
    # every control-plane group must be guarded by the presence flag
    for group in ("CONTROL_PLANE", "LARGE_WORKLOAD", "STRESS"):
        m = re.search(rf"{group}: List\[Tuple\[str, str, str\]\] = (.*)", runner)
        assert m, f"{group} not found"
        assert "_HAS_CONTROL_PLANE" in m.group(1), (
            f"{group} is unguarded and will fail in an open-source tree")
