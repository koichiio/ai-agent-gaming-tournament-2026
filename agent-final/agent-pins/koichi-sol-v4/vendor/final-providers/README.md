# Final Provider Operator Guide

This Python connector implements direct ChatGPT plan usage and explicit
OpenRouter Chat Completions. It does not embed Codex, browser automation, tools,
repository ingestion, or shared conversation state. Official references in the
[integration brief](../plan/chatgpt-provider-integration.md) were rechecked on
October 16, 2026. No operator authentication or live inference was performed
during implementation.

## Fast-Only Policy

The [final-agent plan](../plan/plan.md) requires **Fast mode for every live
inference request**, including development, diagnostics, opponents, and final
validation, at official tournament timing. Do not use Standard or Ultrafast.
Verify Fast support for the selected route/model/account and configure its
documented service setting explicitly. The official
[Fast mode guide](https://developers.openai.com/api/docs/guides/fast-mode.md)
documents `service_tier: "priority"` as equivalent to `"fast"` for supported
models; that mapping alone does not verify this ChatGPT route/account. If
unavailable or downgraded, report it and stop affected new requests rather than
silently changing tiers.

The CLI enforces Fast-only requests, and the adapter factory defaults to
`require_fast: true`. An explicit verified `service_tier` and current capability
evidence are required; omitting them does not select a relaxed default. No Fast
mapping is inferred from a tier name for either provider. Official mapping
evidence is required before a live probe, and successful account-bound probe
validation is required before inference; missing evidence blocks it.

## Install And Check

Run in the project's Nix shell. All commands below are repository-root relative.

```sh
uv sync --project agent-final/providers --locked
agent-final/providers/.venv/bin/python -m pytest agent-final/providers/tests
```

Runnable candidates use their own frozen connector and runtime under
`agent-pins/<candidate>-chatgpt-v1/vendor/`, not this development checkout. Run
`uv sync --project agent-final/agent-pins/<candidate>-chatgpt-v1 --locked` for
each selected copy. Do not use `PYTHONPATH` to connect candidate environments.
Candidate-specific entrypoints and settings are in each copy's `PROVIDER.md`.
These copies retain the committed v1 runtime's 60-second ceiling. The separate
relaxed-timing expansion was cancelled and its timing-only copies removed.

## Select And Sign In

Use a distinct stable local label per account/workspace registration. A label is
not an email or provider account ID. The CLI requires an absolute store beneath
`agent-final/.private/`; never use a global CLI credential file. Replace
`/absolute/project` below with this repository's absolute path.

```sh
agent-final/providers/.venv/bin/final-provider --store /absolute/project/agent-final/.private/chatgpt --account main status
agent-final/providers/.venv/bin/final-provider --store /absolute/project/agent-final/.private/chatgpt --account main --timeout 600 login
```

Login is an explicit operator action. It binds a loopback listener before
opening the system browser. It creates a stable opaque host UUID, fresh state,
nonce, and PKCE S256 values; validates the callback, signature, issuer,
audience, expiry, nonce, and returning identity; and exchanges with the issued
client ID. Reauthorization reuses that ID even after logout.
`--new-registration` is an explicit replacement, not the normal renewal path. An
existing label cannot be overwritten with another subject; choose another label
instead.

If no system browser can open, use
`login --authorization-file authorization-main.txt`. The private file is created
inside the selected store and removed when login exits. Open its URL privately
on this host. Never paste it into diagnostic output, chat, logs, or an issue.
Callback URLs, codes, tokens, and account identifiers are never printed. The
store uses `0700` directories, `0600` files, atomic replacements, and
cross-process locks.

Missing plan-usage scopes keep identity sign-in but disable inference. Repeat
the explicit login action to authorize the complete scope set. There is no
API-key, account, model, or billing fallback.

## Explicit Diagnostics

Status, adapter construction, preflight, and close are network-free. Catalog,
probe, inference, login, and logout contact the provider only when explicitly
requested. Diagnostics do not start a Werewolf match.

```sh
agent-final/providers/.venv/bin/final-provider --store /absolute/project/agent-final/.private/chatgpt --account main catalog
```

Only after verifying an official Fast mapping, use `probe` with both
`--probe-service-tier <verified-tier>` and
`--fast-evidence-file /absolute/project/agent-final/.private/fast-evidence.json`.
The evidence must be an owner-only (`0600`), non-symlink JSON file with exactly
`route`, `model`, `service_tier`, `reference_url`, and `statement`. The first
three must match the request. The reference must use HTTPS on `openai.com` (or a
subdomain) for ChatGPT, or `openrouter.ai` (or a subdomain) for OpenRouter; the
truthful documented statement must explicitly say Fast. A URL and invented
statement are not evidence of an actual mapping.

Probe configuration must leave `effort` and `service_tier` unset; the probe
flags supply them. Optional `--probe-effort <verified-effort>` must accompany
the Fast tier probe, never replace it. After successful verification, use
`infer --config /absolute/project/agent-final/.private/verified-fast.json`: this
JSON contains RouteConfig fields, including the explicit verified
`service_tier`, not a nested `provider_config` or credentials. Inference loads
the selected label's cached evidence. Never run unset-tier inference.

Catalog checks the exact `gpt-6.1-sol` slug from `GET /v1/models`, with no model
search or substitute. Its account-bound capability cache lasts one hour and does
not prove effort or tier support. A new catalog check clears setting evidence;
repeat the necessary probes afterward. A probe authorizes only that explicit
diagnostic request provisionally. It promotes settings only when the completed
response reports the exact served model and requested tier. Optional effort is
promoted separately only when the exact requested effort is also observed. An
accepted request without observations does not verify the setting. Evidence is
private and bound to registration identity, not just a reusable account label.

The private cache is
`<store>/account-<sha256-of-local-label>.capabilities.json`; the hash uses the
label's UTF-8 bytes, not the provider account identity. Inspect the filename
locally/offline without printing account records. There is no capability export
command. Each label has one cache: probing another route replaces it, so use
separate local labels/stores for separate route evidence. The documented
`priority` alias still requires exact observed `priority`, not `fast` or a
missing tier. Fast availability for this route/account and deployment remain
unverified until the parent's live probe succeeds; do not use Standard or
Ultrafast.

OpenRouter uses a separately provisioned owner-only JSON file containing exactly
`{"api_key":"<secret>"}` beneath `agent-final/.private/`. Do not put that
content in manifests or committed examples. Its diagnostics additionally require
`--route openrouter --model <operator-selected-model>` and
`--key-file /absolute/project/agent-final/.private/openrouter.json`. Use the
same documented Fast evidence/probe gates before `infer --config`; no example
model or tier is an approved live declaration.

The ChatGPT store/account flags are required by the common CLI but this
OpenRouter operation does not load ChatGPT tokens. Opponents need not have a
ChatGPT registration. ChatGPT candidates never need an OpenRouter key.

### One-Use Stopped Diagnostic

`probe_once.py` is an isolated operator exception for the single diagnostic
authorized on October 16, 2026. It is not imported by candidate factories. After
review, the parent operator may invoke it exactly once with the existing main
store and priority evidence:

```sh
agent-final/providers/.venv/bin/python agent-final/providers/probe_once.py \
  --store /home/koichi/Projects/ai-agent-game-tournament-ucla/agent-final/.private/chatgpt \
  --account main \
  --fast-evidence-file /home/koichi/Projects/ai-agent-game-tournament-ucla/agent-final/.private/fast-priority-evidence.json \
  --timeout 60 \
  --confirm-stopped-fast-probe
```

This command has **not** been run during implementation. Confirmation burns
`agent-final/.private/diagnostic-20261016T150718Z.used` atomically before any
network, including credential refresh. The receipt remains consumed on failure;
do not delete it, change its path, or retry. There is no CLI receipt override.
The sole completion requests `gpt-6.1-sol` with `service_tier: priority`,
retains Fast evidence and schema/model/tier checks, and ignores only the
unchanged preexisting downgrade marker for that invocation. New downgrade
failures still fail normally. The original stop is never removed or changed,
including on success, so ordinary CLI and gameplay requests remain blocked.

Output contains only fixed requested settings, allowlisted observed settings,
presence/completion booleans, and a success flag or sanitized error code. It
does not expose response bodies, messages, reasoning, credentials, or
identities.

## Per-Seat Configuration

Adapters accept nested `provider_config`. Paths are explicit, relative to
`agent-final` or absolute beneath its `.private/`, never resolved by upward
dotenv discovery. Each seat owns its policy/history and process.

For ChatGPT, nested `provider_config` selects `route: chatgpt`,
`auth_mode: oauth`, model `gpt-6.1-sol`, `focal: true`, an explicit
`credential_source` store, local `account` label, and verified `service_tier`.
See the
[sanitized per-seat checklist](../../testing-battle-simulator/examples/final-provider-seats.md)
and each copy's `PROVIDER.md` for its strategy and accepted adapter fields.
Koichi retains distinct low/high stage requests unless an explicit provider
effort overrides them. Neel likewise retains each caller's effort and uses the
same configured model for extraction, generation, and scoring. Probe every
effort actually requested, or explicitly configure a verified effort and record
that behavioral change. John, Jin, and Starter use provider defaults unless an
effort is explicitly configured. No tier is silently selected.

For an OpenRouter opponent, set `focal: false`, `route: "openrouter"`,
`auth_mode: "api_key"`, its explicit model and `.private/openrouter.json`
source; omit `account`. Also supply `capabilities_file` pointing to protected
verified evidence, separate from the key JSON. Point this directly to the exact
protected capability cache file produced by the matching OpenRouter CLI probe.
Its route/model/credential source must match; OpenRouter caches set `account` to
`null` as required by the factory. The required CLI local label selects the
cache filename only; no relabeling or export is needed. A GPT opponent may
instead use ChatGPT with `focal: false` and its own explicitly selected
registration. Focal requests require ChatGPT and Sol at every stage. Provider
factory/configuration errors never substitute a strategy.

## Request Semantics

ChatGPT sends bearer-authenticated `POST https://api.openai.com/v1/responses`,
array input, `store: false`, and `stream: true`. Ordered system messages become
developer messages; full seat-authorized history is supplied for every call.
There is no previous response ID, server conversation, tool execution, or shared
context. Success requires `response.completed`, final assistant output, valid
JSON/schema, and the candidate's legal-action validation. Reasoning items,
partial JSON, refusals, incomplete streams, and late output are not game
messages.

The parent reported a narrow live ChatGPT transport deviation on October 16,
2026: `service_tier: "fast"` returned HTTP 400 with an unsupported-tier detail;
the documented `"priority"` alias returned HTTP 200 with SSE containing
`response.completed` but no `Content-Type` header. The connector permits only
that missing-header deviation: the unchanged SSE parser must yield a completed
response, and all output/schema, focal-model, exact Fast-tier, and
total-deadline checks still apply before success. Explicit non-SSE or empty MIME
headers are rejected, even with a completed SSE body. Headerless JSON, plain
text, malformed or partial streams, refusals, and failed streams do not succeed.
Requests remain `stream: true`; there is no JSON or non-streaming fallback.
Candidate legal-action validation is unchanged. This transport observation does
not establish Fast verification; that remains pending the parent's successful
live probe.

The preview rejects temperature and output-token limits. Copy configuration must
explicitly select `output_limit_policy: "omit_unsupported"` where required;
metadata records requested limits and `enforced_max_tokens: null`. This is a
documented semantic difference, not an enforced output cap. Direct connector
calls that explicitly request unsupported fields fail before network access.
OpenRouter retains strict schema, reasoning mapping, temperature/token limits,
and `allow_fallbacks: false`/`require_parameters: true`. Explicit supported
effort values are forwarded, not silently translated; model-specific rejection
surfaces.

One total deadline covers refresh lock waits, auth HTTP, connection, stream,
validation, retries in candidate wrappers, and cleanup. The connector has no
hidden SDK retries. Synchronous candidates retire and drain native workers on
timeout rather than let old mutations enter another decision.

## Refresh And Logout

Seat processes share only the selected protected renewable credential record,
not an inference proxy or policy memory. Each access rechecks registration under
the file lock. Renewal serializes rotating refresh tokens across processes,
persists the replacement atomically, and preserves credentials on transient
failures. Signing metadata is fetched before consuming a refresh grant. Invalid
renewed identities are quarantined privately and require reauthorization; they
are never used for inference. Terminal refresh errors require explicit login.
All old, pending, and renewed tokens are registered with seat redactors.

```sh
agent-final/providers/.venv/bin/final-provider --store /absolute/project/agent-final/.private/chatgpt --account main logout
```

Logout attempts discovery-backed renewable-session revocation, then clears local
tokens while retaining the account/client mapping and host ID. An unconfirmed
remote revocation is reported honestly. Disconnect the app in ChatGPT Settings
if needed; no background browser or account switching occurs.

Usage exhaustion writes a shared local stop marker for the affected
registration. New requests stop without changing account, model, or billing.
Review [ChatGPT Settings, Usage](https://chatgpt.com/settings/usage), then
explicitly run `clear-usage-block --confirm` only when appropriate. No reset
time is guessed. Ordinary transient failures and usage-check unavailability
remain distinct. Subscription monetary cost is unknown/null, never API-priced or
invented as zero.

## Remote Hosts

Follow the official
[self-hosted VM procedure](https://developers.openai.com/siwc/token-sharing-open-source/self-hosted-vms).
Give the VM a distinct stable host ID and transfer only the selected protected
registration through an authenticated secure channel. The VM must own later
refreshes. Do not run one rotating session on multiple machines independently,
copy global CLI/browser credentials, or copy other accounts. VM login/deployment
has not been verified here.

## Verification Boundary

Offline tests use synthetic credentials, mock HTTP/SSE, and network-disabled
startup/close. They cover OAuth validation, account binding, atomic storage,
multiprocess rotation, terminal/transient recovery, routing, deadlines,
cancellation, refusals, truncation, schema validation, and installed artifacts.
They establish no account entitlement, actual browser login, provider
throughput, Fast mapping, VM behavior, or game strength. Commit
code/configuration/locks before any later match; log every attempt and finalized
sanitized evidence under `agent-final/data/` as required by `AGENTS.md`.
