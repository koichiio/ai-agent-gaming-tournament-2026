"""Offline launcher tests. No matches, platform calls, or model requests."""

import asyncio
import importlib.util
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

PATH = Path(__file__).resolve().parents[1] / "tournament.py"
SPEC = importlib.util.spec_from_file_location("final_tournament", PATH)
tournament = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tournament)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("Network prohibited in launcher tests")

    monkeypatch.setattr("socket.socket.connect", denied)
    monkeypatch.setattr("socket.getaddrinfo", denied)


def test_owner_spec_and_submission():
    spec = tournament.load_spec()
    assert spec.workspace == tournament.OWNER
    assert spec.agent == spec.name == "final"
    assert spec.command == (str(tournament.PYTHON), "-B", "-m", "agent_koichi.adapter")
    assert spec.config["strategy"] == "hard"
    assert spec.config["planner_enabled"] is False
    assert spec.config["max_chats"] == 2
    assert spec.config["use_provider_default_effort"] is True
    assert spec.config["provider_config"]["require_fast"] is False
    assert spec.config["provider_config"]["account"] == "main"
    assert "model_key" not in spec.config


def test_check_precedes_platform_key_and_commit_gate(monkeypatch):
    version = Mock(return_value="a" * 40)
    preflight = AsyncMock()
    key = Mock(side_effect=AssertionError("Must not load platform key"))
    committed = Mock(
        side_effect=AssertionError("Check permits uncommitted owner files")
    )
    monkeypatch.setattr(tournament, "running_version", version)
    monkeypatch.setattr(tournament, "preflight", preflight)
    monkeypatch.setattr(tournament, "official_key", key)
    monkeypatch.setattr(tournament, "committed_owner", committed)
    tournament.main(["--check"])
    version.assert_called_once_with(tournament.SOURCE, require_clean=False)
    spec, sha = preflight.call_args.args
    assert sha == "a" * 40
    assert spec.command[:3] == (str(tournament.PYTHON), "-B", "-c")
    assert "sys.addaudithook(offline)" in spec.command[3]
    key.assert_not_called()
    committed.assert_not_called()


@pytest.mark.parametrize("argv", [[], ["--match"], ["--check", "--tournament"]])
def test_explicit_mode_required(argv):
    with pytest.raises(SystemExit):
        tournament.main(argv)


def test_owner_version_bridge_and_restore(monkeypatch):
    original = tournament.runtime.running_version
    committed = Mock()
    source_version = Mock(return_value="a" * 40)

    async def supervise(lineup, key, versions, runner_version, **kwargs):
        assert tournament.runtime.running_version(tournament.OWNER) == "a" * 40
        assert kwargs == dict(
            max_matches=None, runtime_workspace=tournament.SOURCE, keep_alive=True
        )
        raise RuntimeError("synthetic stop")

    monkeypatch.setattr(tournament, "committed_owner", committed)
    monkeypatch.setattr(tournament, "running_version", source_version)
    monkeypatch.setattr(tournament.runtime, "supervise", supervise)
    with pytest.raises(RuntimeError, match="synthetic stop"):
        asyncio.run(tournament.supervise_owner(None, "synthetic-key", "a" * 40))
    assert tournament.runtime.running_version is original
    committed.assert_called_once_with("a" * 40)
    source_version.assert_called_once_with(tournament.SOURCE, require_clean=True)


def test_tournament_routes_no_testing_assignments():
    spec = tournament.load_spec()
    lineup = tournament.StandaloneLineup(spec, testing=False, tournament=True)
    assignment = dict(game_type="werewolf", match_status="starting", seat_position=0)
    assert lineup.route(assignment | {"context": "testing"}) is None
    assert lineup.route(assignment | {"context": "tournament"}) is spec


def test_uncommitted_owner_bytes_rejected_before_key(monkeypatch):
    monkeypatch.setattr(tournament, "running_version", Mock(return_value="a" * 40))
    key = Mock(side_effect=AssertionError("Must reject before platform key"))
    monkeypatch.setattr(tournament, "official_key", key)
    monkeypatch.setattr(
        tournament.subprocess,
        "run",
        Mock(side_effect=[Mock(stdout="agent-final/"), Mock(returncode=1, stdout=b"")]),
    )
    with pytest.raises(tournament.ConfigurationError, match="Commit the exact"):
        tournament.main(["--tournament"])
    key.assert_not_called()


def test_live_launch_uses_one_owner_lock_and_existing_supervisor(monkeypatch):
    sha = "a" * 40
    version = Mock(return_value=sha)
    committed = Mock()
    validate = Mock()
    key = Mock(return_value="synthetic-key")
    lock = Mock()
    lock.fileno.return_value = 17
    opened = Mock(return_value=nullcontext(lock))
    flock = Mock()
    supervise = AsyncMock()
    monkeypatch.setattr(tournament, "running_version", version)
    monkeypatch.setattr(tournament, "committed_owner", committed)
    monkeypatch.setattr(tournament, "validate_experiments", validate)
    monkeypatch.setattr(tournament, "official_key", key)
    monkeypatch.setattr(tournament, "_directory", Mock())
    monkeypatch.setattr(tournament, "_open", opened)
    monkeypatch.setattr(tournament.fcntl, "flock", flock)
    monkeypatch.setattr(tournament, "supervise_owner", supervise)
    tournament.main(["--tournament"])
    version.assert_called_once_with(tournament.SOURCE, require_clean=True)
    committed.assert_called_once_with(sha)
    key.assert_called_once_with(tournament.OWNER)
    assert opened.call_count == 1
    assert opened.call_args.args[0].parent == tournament.OWNER / ".private"
    flock.assert_called_once_with(
        17, tournament.fcntl.LOCK_EX | tournament.fcntl.LOCK_NB
    )
    lineup, passed_key, passed_version = supervise.call_args.args
    assert passed_key == "synthetic-key" and passed_version == sha
    assert lineup.testing is False and lineup.tournament is True
    validate.assert_called_once_with(lineup, {"final": sha})


def test_experiment_conflict_rejected_before_credentials(monkeypatch):
    monkeypatch.setattr(tournament, "running_version", Mock(return_value="a" * 40))
    monkeypatch.setattr(tournament, "committed_owner", Mock())
    monkeypatch.setattr(
        tournament,
        "validate_experiments",
        Mock(side_effect=ValueError("new experiment ID")),
    )
    key = Mock()
    monkeypatch.setattr(tournament, "official_key", key)
    with pytest.raises(ValueError, match="new experiment ID"):
        tournament.main(["--tournament"])
    key.assert_not_called()
