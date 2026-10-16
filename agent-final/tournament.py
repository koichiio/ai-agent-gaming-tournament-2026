"""Owner-level tournament launcher; run with the selected candidate's interpreter."""

import argparse
import asyncio
import fcntl
import hashlib
import re
import subprocess
import sys
import tomllib
from dataclasses import fields, replace
from pathlib import Path

from agent_koichi.config import WORKSPACE as INSTALLED_SOURCE
from agent_koichi.config import Config, ConfigurationError
from agent_koichi.standalone import StandaloneLineup, preflight
from werewolf_runtime import runtime
from werewolf_runtime.audit import Redactor, _directory, _open
from werewolf_runtime.config import (
    AgentSpec,
    official_key,
    running_version,
    validate_experiments,
)

OWNER = Path(__file__).resolve().parent
SOURCE = OWNER / "agent-pins" / "koichi-sol-v4"
PYTHON = SOURCE / ".venv" / "bin" / "python"
CONFIG = OWNER / "submission.toml"

# Audit hooks are installed only in the disposable check adapter process.
OFFLINE_ADAPTER = """
import runpy, sys
def offline(event, args):
    if event in ('socket.connect', 'socket.getaddrinfo', 'socket.bind', 'socket.sendto', 'socket.sendmsg'):
        raise RuntimeError('Network disabled during submission check')
sys.addaudithook(offline)
runpy.run_module('agent_koichi.adapter', run_name='__main__')
"""


def load_spec() -> AgentSpec:
    if INSTALLED_SOURCE.resolve() != SOURCE.resolve():
        raise ConfigurationError(
            "Use koichi-sol-v4/.venv/bin/python for this launcher."
        )
    data = tomllib.loads(CONFIG.read_text(encoding="utf-8"))
    allowed = {entry.name for entry in fields(Config)} - {
        "workspace",
        "model_key",
        "baseline_version",
    }
    if (
        set(data) != {"config", "experiment"}
        or not isinstance(data["config"], dict)
        or set(data["config"]) - allowed
        or Redactor([]).clean(data) != data
    ):
        raise ConfigurationError("Expected non-secret config and experiment tables.")
    experiment = data["experiment"]
    if (
        not isinstance(experiment, dict)
        or set(experiment)
        != {"experiment_id", "baseline_version", "changes", "hypothesis"}
        or any(not isinstance(value, str) for value in experiment.values())
        or any(
            not experiment[key].strip()
            for key in ("experiment_id", "changes", "hypothesis")
        )
        or (
            experiment["baseline_version"]
            and not re.fullmatch(r"[0-9a-f]{40}", experiment["baseline_version"])
        )
        or not data["config"].get("strategy")
    ):
        raise ConfigurationError(
            "Supply explicit strategy and valid experiment metadata."
        )
    config = Config.from_env(
        **data["config"],
        workspace=SOURCE,
        model_key="",
        baseline_version=experiment["baseline_version"] or None,
    )
    config.require_model()
    return AgentSpec(
        name="final",
        workspace=OWNER,
        agent="final",
        command=(str(PYTHON), "-B", "-m", "agent_koichi.adapter"),
        strategy=config.strategy,
        config={
            entry.name: getattr(config, entry.name)
            for entry in fields(Config)
            if entry.name not in ("workspace", "model_key")
        },
        experiment=experiment,
    )


def committed_owner(version: str) -> None:
    """Check only effective owner code/config, not unrelated integration files."""
    prefix = subprocess.run(
        ["git", "rev-parse", "--show-prefix"],
        cwd=OWNER,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    for path in (Path(__file__).resolve(), CONFIG):
        committed = subprocess.run(
            ["git", "show", f"{version}:{prefix}{path.name}"],
            cwd=OWNER,
            capture_output=True,
            check=False,
        )
        if committed.returncode or committed.stdout != path.read_bytes():
            raise ConfigurationError(
                "Commit the exact tournament.py and submission.toml before launching."
            )


async def supervise_owner(lineup, key: str, version: str) -> None:
    original = runtime.running_version

    def selected_version(workspace, *, require_clean=True):
        if Path(workspace).resolve() == OWNER:
            committed_owner(version)
            workspace = SOURCE
        return running_version(workspace, require_clean=require_clean)

    # The existing supervisor checks spec.workspace, which intentionally owns records
    # rather than source here. Preserve its checks with this narrowly scoped bridge.
    runtime.running_version = selected_version
    try:
        await runtime.supervise(
            lineup,
            key,
            {"final": version},
            version,
            max_matches=None,
            runtime_workspace=SOURCE,
            keep_alive=True,
        )
    finally:
        runtime.running_version = original


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--check", action="store_true", help="Offline adapter startup/close only"
    )
    mode.add_argument(
        "--tournament",
        action="store_true",
        help="Poll authorized tournament assignments only",
    )
    args = parser.parse_args(argv)
    spec = load_spec()
    version = running_version(SOURCE, require_clean=not args.check)
    if args.check:
        offline = replace(spec, command=(str(PYTHON), "-B", "-c", OFFLINE_ADAPTER))
        asyncio.run(preflight(offline, version))
        return
    committed_owner(version)
    lineup = StandaloneLineup(spec, testing=False, tournament=True)
    validate_experiments(lineup, {spec.name: version})
    key = official_key(OWNER)
    private = OWNER / ".private"
    _directory(private)
    with _open(
        private / (hashlib.sha256(key.encode()).hexdigest() + ".lock"), "a+"
    ) as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        asyncio.run(supervise_owner(lineup, key, version))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError):
        # Do not expose credential-bearing exceptions or subprocess diagnostics.
        raise SystemExit(
            "Submission startup failed; check committed source/config, local credentials, and supervisor lock."
        ) from None
