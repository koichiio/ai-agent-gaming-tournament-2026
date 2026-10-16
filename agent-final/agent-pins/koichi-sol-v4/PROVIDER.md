# Koichi Sol V3

Clean final-owned source copy of `koichi-sol-v2`. Prior copies are untouched.
Agent distribution/module names remain `agent-koichi` and `agent_koichi`. The
execution identity is `koichi-sol-v3`; use its own `.venv/bin/python` with
`-m agent_koichi.adapter`. Worker context must identify this canonical copy as
`workspace`, `agent: final`, and its owning `agent-final` as `record_workspace`.

## Ready Configuration

The user explicitly authorized default service gameplay for this candidate.
Select `hard`, `energy`, or `csp`; each accepts `planner_enabled: true` or
`false`. All stages share the selected Sol/sign-in client.

```json
{
  "strategy": "csp",
  "planner_enabled": true,
  "model": "gpt-6.1-sol",
  "use_provider_default_effort": true,
  "output_limit_policy": "omit_unsupported",
  "provider_config": {
    "route": "chatgpt",
    "auth_mode": "oauth",
    "model": "gpt-6.1-sol",
    "focal": true,
    "require_fast": false,
    "credential_source": ".private/chatgpt",
    "account": "operator-selected-label",
    "capabilities_file": ".private/verified-capabilities.json"
  }
}
```

The parent must substitute its existing authorized account and capability file.
Do not specify `effort` or `service_tier`. Account/source-bound valid Sol
catalog capabilities are still required; low/high or Fast entitlement is not
inferred. `use_provider_default_effort` defaults to false and rejects
nonbooleans and conflicting explicit effort. When true the facade passes
`effort=None` for extraction and both policies. Effective effort is logged as
`provider_default`, requested effort is null, and original low/high remains
`policy_effort`. Observed effort is only provider-reported, never invented.
Omitted service tier is requested null; actual served tier (such as `default`)
is recorded separately.

Factory `require_fast` defaults to true and rejects nonbooleans. Only explicit
false permits non-Fast inference. Fast configurations retain all existing
entitlement, exact-tier and persistent stop checks. No real stop marker,
diagnostic receipt, credentials, or one-use probe was accessed or changed here.
The diagnostic CLI remains Fast-only and must not be used for this opt-out.

## Offline Closure

`PROVENANCE.json` records the frozen current provider source hashes. Runtime
bytes, dependency declarations and `uv.lock` are inherited unchanged from v2.
Connector/runtime install noneditably from this copy's vendor directories;
candidate code installs editably from this copy's own source.

```sh
UV_CACHE_DIR="$PWD/.uv-cache" UV_PYTHON_DOWNLOADS=never \
  uv sync --locked --offline --link-mode copy
```

Verification uses the own interpreter in isolated mode, disabled plugin
autoload, warnings as errors, and a socket audit hook prohibiting network. See
provenance for results. Offline tests establish correctness, not live
entitlement, timing, strength, or tournament readiness. No live calls, matches,
or commits were run.

## Verification

- Full provider suite: **653 passed**, 11.70 seconds, no failures or skips.
- Full Koichi suite: **3,958 passed**, 39.18 seconds, no failures or skips; own
  isolated interpreter, outside-source working directory, network audit
  blocking, warnings treated as errors.
- Includes 54 actual-factory startup combinations covering Fast and explicit
  default-service settings. ChatGPT default fixtures contain no low/high or Fast
  entitlement evidence. Existing Fast downgrade/stop regressions still pass.
- Installed closure checks verify own imports, noneditable vendor dependencies,
  source URLs, every frozen connector hash, and installed source bytes.
- All 25 installed packages are compatible. Lockfile, dependency declarations,
  and runtime source bytes are unchanged from v2. Scoped formatting check
  passed.

Frozen connector SHA-256:
`2c3cb80089209b79fb9661114e0692293548aae6cae9355da1d7768bfd50c61e`.
