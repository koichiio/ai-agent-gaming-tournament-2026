# Final Werewolf Submission

Finalized Koichi Sol V4, exported without behavior changes from development
commit `43b03f4a73f9cf8a889b4f4d23f5b1d44f93f2a2`. The canonical source remains
`agent-final/agent-pins/koichi-sol-v4`. `SOURCE.json` records exact exported
hashes; `PROVENANCE.json` inside the candidate retains its original upstream
evidence.

## Install And Run

From this repository root, with Nix and uv available:

```sh
nix develop
uv sync --locked --project agent-final/agent-pins/koichi-sol-v4
agent-final/agent-pins/koichi-sol-v4/.venv/bin/python -B agent-final/tournament.py --check
agent-final/agent-pins/koichi-sol-v4/.venv/bin/python -B agent-final/tournament.py --tournament
```

Use the candidate's own interpreter, not another checkout's environment.
`--check` blocks network and performs adapter startup/close only; valid local
provider credentials and capability evidence are still required. `--tournament`
requires committed source/configuration and polls authorized tournament
assignments only. Do not run it before the parent operator completes
provisioning and commits this repository. Run only one supervisor per Official
Agent Key.

Platform connectivity is a separate parent-managed step, not an agent match:

```sh
# Placeholder: parent operator supplies and reviews this script and its arguments.
<parent-managed-platform-connectivity-command>
```

## Local Security

No credentials, keys, OAuth records, private state, match data, or development
history are included. Original provider OAuth sign-in is required locally; no
model API key is bundled or required for the selected ChatGPT route. The parent
operator provisions `ALTRUAGENT_OFFICIAL_AGENT_KEY` in ignored
`agent-final/.env` and the original provider's local OAuth store at ignored
`agent-final/.private/chatgpt`, account label `main`. Account-bound current
model capability evidence is also required; do not fabricate or copy unrelated
records. Never print credentials or publish live journals. This export performs
no login, connectivity request, inference, match, commit, or push.

`agent-final/submission.toml` is authoritative: model `gpt-6.1-sol`, hard
strategy, planner off, two chats, provider-default effort, explicit
`require_fast = false`. The frozen provider README retains historical Fast-only
operator instructions; those are not the selected submission's configuration. Do
not run historical diagnostic commands from that README. No provider, account,
or model fallback is authorized. The parent owns local provisioning and
connectivity verification.

## Offline Checks

```sh
agent-final/agent-pins/koichi-sol-v4/.venv/bin/python -B -m pytest -p no:cacheprovider agent-final/tests agent-final/agent-pins/koichi-sol-v4/tests
nix flake check --no-update-lock-file
```

Tests use synthetic fixtures, not live services. `test_provider_copy.py` creates
synthetic private stores beneath pytest temporary directories. The original
executable `.githooks/pre-commit` runs the real Nix formatting check;
`nix develop` selects that hook without persistent Git configuration. Offline
checks do not establish live entitlement, connectivity, latency, or game
strength.

## Attribution And License

The original root `LICENSE` is preserved verbatim, including its scope
exclusions. Its references to development-only upstream attribution paths do not
imply those excluded components are bundled here. Original frozen provider
documentation and source notices are preserved. No other candidates or
starter-agent code are exported.
