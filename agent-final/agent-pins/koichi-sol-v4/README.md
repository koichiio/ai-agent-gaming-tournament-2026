# Koichi Sol V4

Final Werewolf candidate. Install and launch using the repository-root README
and `agent-final/tournament.py` with this candidate's own `.venv/bin/python`.
`agent-final/submission.toml` is authoritative: hard strategy, planner disabled,
two chats, `gpt-6.1-sol`, original-provider OAuth, provider-default effort, and
explicit non-Fast opt-in. Local OAuth sign-in and current account-bound
capability evidence are required; no credentials or API key are distributed.

All Python source, dependency declarations, locks, `PROVENANCE.json`, and
`PROVIDER.md` are committed bytes from the development SHA in root
`SOURCE.json`. This minimal README replaces development documentation only.
`PROVIDER.md` and the frozen vendor README retain historical details; the root
README and selected submission configuration supersede historical copy names,
paths, and Fast-only deployment instructions. Original vendor documentation is
preserved byte-for-byte because its hash is part of the frozen connector
provenance.

The local dependency closure is `vendor/final-providers` and
`vendor/werewolf-runtime`; no development checkout or other candidate is
required. Selected self-contained offline tests are included. See root
`SOURCE.json` for test omissions and exact exported hashes. No live validation
or performance claim is implied by this export. The root `LICENSE` and its scope
exclusions apply.
