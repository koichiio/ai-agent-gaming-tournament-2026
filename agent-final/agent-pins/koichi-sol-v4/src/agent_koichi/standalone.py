"""Independent official-protocol runtime, with no Testing supervisor dependency."""

import asyncio
import fcntl
import hashlib
import re
import subprocess
import sys
import tomllib
from dataclasses import fields
from pathlib import Path

from werewolf_runtime.audit import Redactor, _directory, _open
from werewolf_runtime.config import (
    AgentSpec,
    official_key,
    running_version,
    validate_experiments,
)
from werewolf_runtime.protocol import VERSION, validate_metadata
from werewolf_runtime.runtime import context_for, supervise
from werewolf_runtime.worker import Worker

from .config import WORKSPACE, Config, ConfigurationError


class StandaloneLineup:
    """One policy for every authorized seat in the explicitly selected contexts."""

    def __init__(self, spec: AgentSpec, *, testing: bool, tournament: bool) -> None:
        self.agents = {spec.name: spec}
        self.spec, self.testing, self.tournament = spec, testing, tournament

    def route(self, assignment: dict) -> AgentSpec | None:
        context = assignment.get("context")
        # The official protocol treats assignments without context as tournament games.
        selected = (
            self.testing
            if context == "testing"
            else self.tournament
            if context in (None, "tournament")
            else False
        )
        if (
            not selected
            or assignment.get("game_type") != "werewolf"
            or assignment.get("match_status") not in ("starting", "in_progress")
            or type(assignment.get("seat_position")) is not int
            or not 0 <= assignment["seat_position"] < 7
        ):
            return None
        return self.spec

    def details(
        self, assignments: list[dict], versions: dict[str, str]
    ) -> tuple[str, list]:
        details = [
            {
                "seat_position": assignment["seat_position"],
                "agent": self.spec.agent,
                "version": versions[self.spec.name],
                "strategy": self.spec.strategy,
            }
            for assignment in assignments
            if self.route(assignment) is not None
        ]
        kind = (
            "homogeneous"
            if {row["seat_position"] for row in details} == set(range(7))
            else "unknown"
        )
        return kind, details


def load_run(path: Path, overrides: dict) -> AgentSpec:
    path = path.resolve(strict=True)
    if not path.is_relative_to(WORKSPACE):
        raise ConfigurationError("Run configuration must be inside agent-koichi/.")
    data = tomllib.loads(path.read_text(encoding="utf-8"))
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
        raise ConfigurationError(
            "Run configuration requires non-secret config and experiment tables."
        )
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
        or not isinstance(data["config"].get("strategy"), str)
        or not data["config"]["strategy"].strip()
    ):
        raise ConfigurationError(
            "Supply an explicit strategy, experiment ID, changes, hypothesis, and baseline SHA (or empty)."
        )
    config = Config.from_env(
        **(
            data["config"]
            | {key: value for key, value in overrides.items() if value is not None}
        ),
        baseline_version=experiment["baseline_version"] or None,
    )
    config.require_model()
    # Freeze non-secret settings for all seat processes; each loads its own model key.
    seat_config = {
        entry.name: getattr(config, entry.name)
        for entry in fields(Config)
        if entry.name not in ("workspace", "model_key")
    }
    return AgentSpec(
        name="koichi",
        workspace=WORKSPACE,
        agent="koichi",
        command=(sys.executable, "-m", "agent_koichi.adapter"),
        strategy=config.strategy,
        config=seat_config,
        experiment=experiment,
    )


async def preflight(spec: AgentSpec, version: str) -> None:
    worker = Worker(spec, lambda event, data: None)
    try:
        await worker.launch()
        metadata = validate_metadata(
            await worker.request(
                "start",
                {
                    "protocol_version": VERSION,
                    "context": context_for(
                        spec,
                        {
                            "match_id": "preflight-not-a-match",
                            "seat_id": "preflight-not-a-seat",
                            "seat_position": 0,
                        },
                        version,
                        version,
                    ),
                    "config": spec.config,
                },
            )
        )
        if metadata["strategy"] != spec.strategy:
            raise ConfigurationError(
                "Adapter strategy differs from the standalone configuration."
            )
    finally:
        await worker.close()
    print("Koichi standalone adapter: ready (no platform calls or inference).")


def run(args) -> None:
    if not args.run_config or not (args.match or args.tournament):
        raise ConfigurationError(
            "play requires --run-config and --match and/or --tournament."
        )
    if args.max_matches is not None and args.max_matches < 1:
        raise ConfigurationError("max-matches must be positive.")
    if args.baseline_version is not None:
        raise ConfigurationError(
            "Set baseline_version in the committed run configuration."
        )
    path = Path(args.run_config)
    if not path.is_absolute():
        path = WORKSPACE / path
    spec = load_run(
        path,
        {
            "model": args.model,
            "strategy": args.strategy,
            "planner_enabled": args.planner_enabled,
            "extraction_budget_policy": args.extraction_budget_policy,
        },
    )
    version = running_version(WORKSPACE, require_clean=not args.check)
    if args.check:
        asyncio.run(preflight(spec, version))
        return
    # Reject ignored/untracked run files too: every effective non-secret setting is reproducible.
    relative = str(path.resolve().relative_to(WORKSPACE))
    prefix = subprocess.run(
        ["git", "rev-parse", "--show-prefix"],
        cwd=WORKSPACE,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    committed = subprocess.run(
        ["git", "show", f"{version}:{prefix}{relative}"],
        cwd=WORKSPACE,
        capture_output=True,
        check=True,
    )
    if committed.stdout != path.read_bytes():
        raise ConfigurationError("Commit the exact run configuration before playing.")
    lineup = StandaloneLineup(spec, testing=args.match, tournament=args.tournament)
    versions = {spec.name: version}
    validate_experiments(lineup, versions)
    key = official_key(WORKSPACE)
    private = WORKSPACE / ".private"
    _directory(private)
    with _open(
        private / (hashlib.sha256(key.encode()).hexdigest() + ".lock"), "a+"
    ) as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        asyncio.run(
            supervise(
                lineup,
                key,
                versions,
                version,
                max_matches=args.max_matches,
                match_id=args.match_id.removeprefix("playground:")
                if args.match_id
                else None,
                runtime_workspace=WORKSPACE,
                keep_alive=args.max_matches is None,
            )
        )
