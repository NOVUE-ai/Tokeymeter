# ENGINE_MAP.md — K2 physical layout (W1)

Every old import path remains valid forever via identity alias shims
(`sys.modules[old] is canonical`). Canonical homes:

- **engines/execution/**: integrations/{openai, anthropic, openai_async, universal}
- **engines/optimization/**: router cascade compression safe_compress query_compress salience semantic semantic_eval semantic_verify compression_eval cache_optimize cache_admin prompt_ir warmup storage envelope redis_store
- **engines/reliability/**: degraded _fidelity overhead
- **engines/economics/**: pricing usage savings keys reporting reconcile
- **engines/governance/**: policy privacy identity decision content/secrets
- **engines/knowledge/**: memory context_passport
- **engines/trust/**: audit/{log, proof, signers} integrity cipher

Kernel-core (stays top-level): `__init__ __main__ _api decorator events utils paths admin demo metrics runtime/`

Split packages: `integrations/reconcile` → economics; `backends/cipher` → trust; `backends/redis_store` → optimization.
