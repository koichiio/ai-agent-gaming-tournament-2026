"""Workspace commands; new online attempts, unlike recovery, require committed code."""

import argparse
import asyncio
import json
from uuid import UUID

from werewolf_runtime.config import official_key, running_version

from . import config as config_module
from .config import WORKSPACE, Config, ConfigurationError


def committed_version() -> str:
    try:
        return running_version(WORKSPACE)
    except ValueError as error:
        raise ConfigurationError(str(error)) from None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Koichi's exact-belief Werewolf agent", allow_abbrev=False
    )
    parser.add_argument(
        "command",
        choices=(
            "play",
            "online-smoke",
            "config-check",
            "replay",
            "inspect",
            "correct-diagnostic-mode",
            "correct-diagnostic",
            "validate-logs",
            "recover",
        ),
    )
    parser.add_argument(
        "--strategy", choices=("hard", "energy", "csp", "assertion", "llm")
    )
    parser.add_argument(
        "--planner",
        action=argparse.BooleanOptionalAction,
        dest="planner_enabled",
        default=None,
        help="Enable shallow action selection; --no-planner restores the LLM policy.",
    )
    parser.add_argument(
        "--extraction-budget-policy", choices=("fraction", "policy_reserve")
    )
    parser.add_argument("--model")
    parser.add_argument("--baseline-version")
    parser.add_argument("--log", type=str)
    parser.add_argument(
        "--run-config", help="Workspace-local committed TOML run configuration."
    )
    parser.add_argument(
        "--match", action="store_true", help="Play owned Testing assignments."
    )
    parser.add_argument(
        "--tournament", action="store_true", help="Play tournament assignments."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Local play preflight without platform calls or inference.",
    )
    parser.add_argument(
        "--max-matches", type=int, help="Optional batch limit; otherwise keep running."
    )
    parser.add_argument("--match-id", help="Optional exact platform match ID.")
    parser.add_argument("--attempt-id", help="Exact original private attempt UUID.")
    parser.add_argument(
        "--player-id", help="Exact owned platform seat ID for recovery."
    )
    parser.add_argument("--confirmed-ended", action="store_true")
    parser.add_argument(
        "--reason", help="Reason for operator termination confirmation."
    )
    args = parser.parse_args()
    if args.command == "recover":
        if any(
            value is not None
            for value in (
                args.strategy,
                args.planner_enabled,
                args.extraction_budget_policy,
                args.model,
                args.baseline_version,
                args.log,
                args.run_config,
                args.max_matches,
            )
        ) or (args.match or args.tournament or args.check):
            parser.error("recover rejects model, configuration, log, and play options.")
        try:
            valid_attempt = str(UUID(args.attempt_id)) == args.attempt_id
        except ValueError, AttributeError, TypeError:
            valid_attempt = False
        if not valid_attempt:
            parser.error("recover requires --attempt-id with an exact canonical UUID.")
        if (
            not args.match_id
            or not args.match_id.removeprefix("playground:").strip()
            or args.match_id.startswith(("local:", "self-play:"))
            or "\0" in args.match_id
            or not args.player_id
            or not args.player_id.strip()
            or "\0" in args.player_id
        ):
            parser.error(
                "recover requires an exact platform --match-id and --player-id."
            )
        if args.confirmed_ended != (args.reason is not None) or (
            args.reason is not None and not args.reason.strip()
        ):
            parser.error("Use --confirmed-ended together with a nonempty --reason.")
    elif (
        args.attempt_id is not None
        or args.player_id is not None
        or args.confirmed_ended
        or args.reason is not None
    ):
        parser.error("Recovery options are supported only with recover.")
    if args.command != "play" and (
        args.run_config is not None
        or args.match
        or args.tournament
        or args.check
        or args.max_matches is not None
        or (args.match_id is not None and args.command != "recover")
    ):
        parser.error("Play options, including --check, are supported only with play.")
    try:
        if args.command == "recover":
            from werewolf_runtime.audit import Redactor
            from werewolf_runtime.recovery import recover_attempt

            local = config_module.dotenv_values(WORKSPACE / ".env", interpolate=False)
            key = official_key(WORKSPACE)
            redactor = Redactor(
                [
                    local.get("OPENROUTER_API_KEY") or "",
                    config_module.os.environ.get("OPENROUTER_API_KEY") or "",
                    key,
                ]
            )
            result = asyncio.run(
                recover_attempt(
                    WORKSPACE,
                    "koichi",
                    args.attempt_id,
                    args.match_id,
                    args.player_id,
                    key,
                    experiment=None,
                    confirmed_ended=args.confirmed_ended,
                    reason=args.reason,
                    redactor=redactor,
                )
            )
            print(json.dumps(redactor.clean(result), indent=2))
            return
        if args.command == "play":
            from .standalone import run

            run(args)
            return
        if args.command in (
            "replay",
            "inspect",
            "correct-diagnostic-mode",
            "correct-diagnostic",
        ):
            from .replay import inspect_log, replay

            if not args.log:
                raise ConfigurationError(
                    "This command requires --log data/logs/<finalized-log>.json."
                )
            if args.command in ("correct-diagnostic-mode", "correct-diagnostic"):
                from .audit import correct_diagnostic_mode

                result = correct_diagnostic_mode(WORKSPACE / args.log, WORKSPACE)
            else:
                result = (replay if args.command == "replay" else inspect_log)(
                    WORKSPACE / args.log, WORKSPACE
                )
            print(json.dumps(result, indent=2))
            return
        if args.command == "validate-logs":
            from werewolf_runtime.audit import Redactor, validate_public_logs

            environment = {
                **config_module.dotenv_values(WORKSPACE / ".env", interpolate=False),
                **config_module.os.environ,
            }
            try:
                platform_key = official_key(WORKSPACE)
            except ValueError:
                platform_key = ""
            result = validate_public_logs(
                WORKSPACE,
                Redactor([environment.get("OPENROUTER_API_KEY") or "", platform_key]),
            )
            print(json.dumps(result, indent=2))
            if not result["ok"]:
                raise SystemExit(1)
            return
        config = Config.from_env(
            strategy=args.strategy,
            planner_enabled=args.planner_enabled,
            extraction_budget_policy=args.extraction_budget_policy,
            model=args.model,
            baseline_version=args.baseline_version,
        )
        if args.command == "config-check":
            try:
                official_present = bool(official_key(WORKSPACE))
            except ValueError:
                official_present = False
            print(
                json.dumps(
                    {
                        "model_credential_present": bool(config.model_key),
                        "official_credential_present": official_present,
                        "configuration": config.metadata(),
                    },
                    indent=2,
                )
            )
        else:
            config.require_model()
            version = committed_version()
            from .smoke import online_smoke

            result = asyncio.run(online_smoke(config, version))
            print(json.dumps(result, indent=2))
            if result["status"] != "completed":
                raise SystemExit(1)
    except KeyboardInterrupt:
        parser.exit(130, "Agent interrupted; active match traces remain private.\n")
    except ConfigurationError as error:
        if args.command == "recover":
            parser.exit(
                1, "Recovery stopped; retain private audit journals for diagnosis.\n"
            )
        parser.exit(2, f"Configuration error: {error}\n")
    except Exception:
        # Never print exception reprs that may contain a request or bearer header.
        parser.exit(
            1,
            "Recovery stopped; verify original attempt/seat, credentials, official termination, and audit integrity; traces remain private.\n"
            if args.command == "recover"
            else "Agent stopped on an unexpected error; retain private audit journals for diagnosis.\n",
        )
