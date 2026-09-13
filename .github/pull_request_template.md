## What this changes

<!-- One or two sentences. -->

## Checklist

- [ ] `pytest` passes
- [ ] `python scripts/run_all_checks.py` passes (17/17)
- [ ] If this adds or changes a **control**: it works on all four execution
      routes — sync decorator, async decorator, `cache_stream`, and the runtime
      kernel. A control that exists on one path is only as strong as the path a
      team happens to pick.
- [ ] If this adds or changes a **control**: `simulate` was updated in the same
      commit, so `plan` does not report that a real rule does nothing.
- [ ] No prompt or response text can reach the ledger as a result of this change.
- [ ] `CHANGELOG.md` updated if the behaviour is user-visible.
