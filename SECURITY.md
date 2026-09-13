# Security

## Reporting a vulnerability

Please don't open a public issue. Email the maintainer and you'll get an
acknowledgement within a few days.

## What this package does and doesn't do

Worth knowing before you look, because it narrows the surface considerably:

- **No outbound network calls.** Nothing in the package contacts anything. There
  is no telemetry, no license check, no phone-home.
- **No runtime dependencies.** The base install pulls in nothing, and CI fails if
  that ever stops being true.
- **No credentials handled.** Your provider keys go to your provider SDK, not
  through us. `register_key` stores a *label* and a budget, never a secret.
- **No prompt or response text stored.** 26 ledger fields: identifiers, counts,
  money and hashes. Responses are hashed for the progress signal and discarded.
- **Fails open.** If metering breaks, traffic flows. The single exception is
  compliance enforcement, which fails closed — and narrowly: a ruleset that fails
  to *resolve* contributes nothing, so a bug degrades to "no policy", never to
  "deny everything".

## The parts that are actually sensitive

**The policy file is executable authority over your agents.** Write access to it
means the ability to raise a spend ceiling, turn off a model allowlist, or
disable enforcement estate-wide. Treat it like any other control: in version
control, under CODEOWNERS, reviewed.

**The ledger is not secret but it is revealing.** It contains no content, but
task ids, agent names and spend patterns describe your operations. Give it the
same treatment as an application log.

**Halt handlers run in your request path.** A handler you register with
`on_halt` executes inside the caller's thread. We wrap every one — exceptions
are swallowed and counted, and a handler failing repeatedly is retired — but the
code inside it is yours.

## Verifying what you installed

```bash
python -c "from tokeymeter.engines.trust.integrity import verify_self; print(verify_self().status)"
tokeymeter doctor
```

The package ships a manifest of file hashes and checks itself against it. Tamper
detection catches a single injected line.
