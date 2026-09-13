# Contributing

Issues and pull requests are welcome. The most valuable report you can file is
**a working agent that got stopped** — there's an issue template for it, and it's
treated as a priority bug rather than a tuning question.

## Running the tests

```bash
pip install -e ".[dev]"
pytest                              # 1,689 tests
python scripts/run_all_checks.py    # the 17-check release gate
```

The gate checks things the test suite can't: that the integrity manifest matches
the shipped files, that CLI output is ASCII-only (it runs on a legacy Windows
console as often as a Linux terminal), that no placeholder text shipped, and that
the README's quickstart block still executes.

## Two rules that aren't obvious

**Every control must work on all four execution routes** — sync decorator, async
decorator, `cache_stream`, and the runtime kernel. A control that exists on one
path is only as strong as the path a team happens to pick. Three separate bugs of
exactly this shape have been found and fixed; the probe is now standard.

**Enforcement and simulation move together.** If you add a control, add it to
`simulate` in the same commit. Otherwise `tokeymeter plan` reports that a real
rule does nothing, someone applies it believing that, and production halts. This
has happened three times and each one was caught by an audit rather than a test.

## Things that will get a change rejected

- **Anything that puts prompt or response text into a record.** The ledger is
  content-blind by construction, and that is not negotiable — it is the reason
  this can run inside a hospital.
- **A runtime dependency.** The base install has none, CI enforces it, and a
  security reviewer checks it first. Optional extras are fine.
- **An outbound network call.** Same reason.
- **A new default that changes behaviour for existing users.** Ceilings are
  opt-in; enforcement is off unless asked for.
- **A metric that can be wrong without saying so.** If a number can't be
  computed honestly, return `None` and state why. A metric that lies is worse
  than a missing metric — that principle has been re-learned here four times.

## Style

Explain *why* in comments, not *what*. The code says what it does; the comment
should say what would break if it were written the obvious way instead. Most of
the comments in this codebase exist because a specific test caught a specific
mistake, and they say so.
