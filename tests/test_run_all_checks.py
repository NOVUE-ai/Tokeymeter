"""
P0.2 — cross-platform unified check runner.

Verifies the runner is portable by construction: PYTHONPATH joined with the
platform separator, UTF-8 forced for child processes, transient state redirected
into a controlled workspace, and a real end-to-end invocation passing on a fast
suite. No shell or Unix-only commands are used anywhere in the runner.

    PYTHONPATH=. python -m pytest tests/test_run_all_checks.py -q
"""
import importlib.util
import os
import subprocess
import sys


def _load_runner():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    path = os.path.join(root, "scripts", "run_all_checks.py")
    spec = importlib.util.spec_from_file_location("run_all_checks", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, root


def test_repo_root_resolves_to_novue():
    mod, root = _load_runner()
    # the runner's ROOT must be the repo root that contains both packages
    assert os.path.isdir(os.path.join(mod.ROOT, "tokeymeter"))
    # integrations/tokenet ships only in the private tree; the open-source
    # distribution is deliberately Tokeymeter alone.
    if os.path.isdir(os.path.join(mod.ROOT, "integrations")):
        assert os.path.isdir(os.path.join(mod.ROOT, "integrations", "tokenet"))
    assert mod.ROOT == root


def test_child_env_is_cross_platform():
    mod, _ = _load_runner()
    env = mod.child_env("/some/workspace")
    # PYTHONPATH starts with repo root, joined using the platform separator
    assert env["PYTHONPATH"].startswith(mod.ROOT)
    if "PYTHONPATH" in os.environ and os.environ["PYTHONPATH"]:
        assert os.pathsep in env["PYTHONPATH"]
    # UTF-8 forced for children (Windows-safe output)
    assert env["PYTHONUTF8"] == "1"
    assert env["PYTHONIOENCODING"] == "utf-8"
    # local state redirected into the controlled workspace
    assert env["TOKENET_WORKSPACE"] == "/some/workspace"
    assert env["TOKEYMETER_HOME"].startswith("/some/workspace")
    # every check's tempfiles land under the workspace too (TMPDIR/TEMP/TMP)
    assert env["TMPDIR"].startswith("/some/workspace")
    assert env["TEMP"] == env["TMPDIR"] and env["TMP"] == env["TMPDIR"]


def test_no_shell_or_unix_commands_in_runner():
    import re
    mod, root = _load_runner()
    src = open(os.path.join(root, "scripts", "run_all_checks.py")).read()
    # portability guards: never shell out, never assume a unix-only binary
    assert "shell=True" not in src
    assert "os.system" not in src
    for unixism in ("/tmp/", "rm -rf", "python3 "):
        assert unixism not in src, f"non-portable token in runner: {unixism!r}"
    # the unix `timeout <n> cmd` command (not the word "timeout" in help/kwargs)
    assert not re.search(r"\btimeout\s+\d", src), "uses the unix timeout command"
    # must use the running interpreter + platform-aware separator + subprocess timeout
    assert "sys.executable" in src
    assert "os.pathsep" in src
    assert "timeout=" in src  # cross-platform timeout via subprocess


def test_runner_end_to_end_on_fast_suite():
    """Run the runner for real on one fast suite; it must pass and clean up.

    Uses a check that exists in EVERY tree. The previous target was a control
    plane suite, which is absent from the open-source distribution — so this
    test failed on exactly the code that ships.
    """
    _, root = _load_runner()
    env = dict(os.environ)
    env["PYTHONPATH"] = root + (os.pathsep + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else "")
    proc = subprocess.run(
        [sys.executable, os.path.join(root, "scripts", "run_all_checks.py"),
         "--only", "selftest", "--timeout", "120"],
        cwd=root, env=env, capture_output=True, encoding="utf-8", errors="replace", timeout=180)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "1/1 checks passed" in proc.stdout
    assert "workspace cleaned" in proc.stdout
