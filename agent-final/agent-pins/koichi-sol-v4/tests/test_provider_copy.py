"""API-free facade fixtures. No authentication, inference service, or matches."""

import asyncio
import json
import socket
import time
from dataclasses import asdict
from types import SimpleNamespace

import pytest
from agent_koichi import adapter, model
from agent_koichi.config import Config, ConfigurationError
from agent_koichi.extraction import ExtractionAudit
from final_providers.auth import SCOPES, CredentialStore
from final_providers.cli import _write_record, registration_fingerprint
from final_providers.errors import ProviderError
from final_providers.factory import build_client
from final_providers.inference import Completion, InferenceClient, ModelCapabilities
from werewolf_runtime.audit import Redactor


class Sink:
    def __init__(self):
        self.redactor = Redactor([])
        self.events = []

    def record(self, event, data):
        self.events.append((event, self.redactor.clean(data)))


def provider(route="chatgpt", focal=True):
    return {
        "route": route,
        "auth_mode": "oauth" if route == "chatgpt" else "api_key",
        "model": "gpt-6.1-sol",
        "focal": focal,
        "require_fast": True,
        "credential_source": ".private/synthetic-source",
        **({"account": "synthetic-account"} if route == "chatgpt" else {}),
    }


@pytest.fixture
def clients(monkeypatch, tmp_path):
    monkeypatch.setattr(
        adapter, "WORKSPACE", tmp_path / "agent-final/agent-pins/koichi-sol-v4"
    )
    built = []

    def build(config, workspace, register_secret=None):
        class Client:
            calls = None
            closed = False

            def __init__(self):
                self.calls = []
                self.config = SimpleNamespace(**config)
                self.workspace = workspace
                self.register = register_secret

            async def complete(self, messages, **kwargs):
                self.calls.append((messages, kwargs))
                self.register("synthetic-renewed-access")
                await asyncio.sleep(0)
                return Completion(
                    text='{"ok":true}',
                    parsed={"ok": True},
                    usage={"input_tokens": 3},
                    latency_ms=1,
                    served_model=config["model"],
                    service_tier=None,
                    cost=None if config["route"] == "chatgpt" else 0.02,
                    metadata={},
                )

            async def close(self):
                self.closed = True

        client = Client()
        built.append(client)
        return client

    monkeypatch.setattr(model, "build_client", build)
    monkeypatch.setattr(
        adapter, "resolved_metadata", lambda c: {"provider": c.config.route}
    )
    return built


@pytest.mark.parametrize(
    "route,focal", [("chatgpt", True), ("chatgpt", False), ("openrouter", False)]
)
@pytest.mark.parametrize(
    "strategy,planner",
    [
        ("hard", False),
        ("energy", False),
        ("csp", False),
        ("assertion", False),
        ("llm", False),
        ("hard", True),
        ("energy", True),
        ("csp", True),
        ("assertion", True),
    ],
)
def test_startup_matrix_is_explicit_network_free(
    monkeypatch, tmp_path, clients, route, focal, strategy, planner
):
    for key in ("OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(Config, "runtime_metadata", Config.metadata)
    sink = Sink()
    instance = adapter.KoichiAdapter(sink)
    configuration = {
        "strategy": strategy,
        "planner_enabled": planner,
        "model": "gpt-6.1-sol",
        "provider_config": provider(route, focal),
        "output_limit_policy": "omit_unsupported" if route == "chatgpt" else "enforce",
    }

    async def run():
        metadata = await instance.start(
            {"agent": "final", "workspace": str(tmp_path / "agent-final")},
            configuration,
        )
        assert metadata["strategy"] == strategy
        assert metadata["configuration"]["provider"] == route
        assert (
            metadata["configuration"]["model"]
            == configuration["provider_config"]["model"]
        )
        assert instance.agent.config.planner_enabled is planner
        assert clients[0].calls == []
        await instance.close()
        assert clients[0].closed

    asyncio.run(run())


@pytest.mark.parametrize("route", ["chatgpt", "openrouter"])
def test_auxiliary_and_policy_share_client_envelopes_and_rotated_redaction(
    clients, tmp_path, route
):
    sink = Sink()
    config = Config(
        model="gpt-6.1-sol",
        workspace=tmp_path,
        provider_config=provider(route, route == "chatgpt"),
        output_limit_policy="omit_unsupported" if route == "chatgpt" else "enforce",
    )
    instance = model.ChatModel(config, sink)
    auxiliary = ExtractionAudit("fixture-job", "")

    async def run():
        results = await asyncio.gather(
            instance.complete(
                [{"role": "user", "content": "extract"}], "low", 1, audit=auxiliary
            ),
            instance.complete([{"role": "user", "content": "policy"}], "high", 1),
        )
        assert [r["parsed"] for r in results] == [{"ok": True}, {"ok": True}]
        assert {r["effort"] for r in results} == {"low", "high"}
        assert len({r["call_id"] for r in results}) == 2
        assert all(
            r["provider"] == route and r["served_provider"] is None for r in results
        )
        if route == "chatgpt":
            assert all(r["cost"] is None for r in results)
        for redactor in (sink.redactor, auxiliary.redactor, instance.redactor):
            assert redactor.clean("synthetic-renewed-access") == "[REDACTED]"
        auxiliary.flush(sink)
        assert any(
            event == "model_usage" and data["usage"].get("input_tokens") == 3
            for event, data in sink.events
        )
        for _, kwargs in clients[0].calls:
            assert 0 < kwargs["seconds"] <= 1
            assert "temperature" not in kwargs
            assert ("max_tokens" in kwargs) is (route == "openrouter")
        await instance.close()

    asyncio.run(run())


def test_explicit_effort_is_recorded_not_silently_translated(clients):
    cfg = provider() | {"effort": "medium"}
    instance = model.ChatModel(
        Config(
            model="gpt-6.1-sol",
            provider_config=cfg,
            output_limit_policy="omit_unsupported",
        ),
        Sink(),
    )
    result = asyncio.run(
        instance.complete([{"role": "user", "content": "fixture"}], "high", 1)
    )
    assert result["effort"] == "medium" and result["policy_effort"] == "high"


@pytest.mark.parametrize("policy_effort", ["low", "high"])
def test_provider_default_effort_omits_request_and_preserves_policy(
    clients, policy_effort
):
    sink = Sink()
    config = Config(
        model="gpt-6.1-sol",
        provider_config=provider() | {"require_fast": False},
        output_limit_policy="omit_unsupported",
        use_provider_default_effort=True,
    )
    instance = model.ChatModel(config, sink)
    result = asyncio.run(
        instance.complete([{"role": "user", "content": "fixture"}], policy_effort, 1)
    )
    assert clients[0].calls[0][1]["effort"] is None
    assert result["effort"] == "provider_default"
    assert result["requested_effort"] is None
    assert result["policy_effort"] == policy_effort
    assert result["metadata"].get("observed_effort") is None
    request = next(data for event, data in sink.events if event == "model_request")
    assert request["effort"] == request["payload"]["effort"] == "provider_default"
    assert request["payload"]["requested_effort"] is None
    assert request["policy_effort"] == policy_effort
    assert (
        config.metadata()["fast_effort"]
        == config.metadata()["deep_effort"]
        == "provider_default"
    )


@pytest.mark.parametrize("value", [None, 0, 1, "true", "false"])
def test_provider_default_effort_requires_boolean(value):
    with pytest.raises(ConfigurationError, match="must be a boolean"):
        Config(model="gpt-6.1-sol", use_provider_default_effort=value)
    with pytest.raises(ConfigurationError, match="must be a boolean"):
        Config.from_env(model="gpt-6.1-sol", use_provider_default_effort=value)


def test_provider_default_effort_rejects_explicit_effort_conflict():
    with pytest.raises(ConfigurationError, match="conflicts"):
        Config(
            model="gpt-6.1-sol",
            provider_config=provider() | {"effort": "high"},
            output_limit_policy="omit_unsupported",
            use_provider_default_effort=True,
        )


@pytest.mark.parametrize(
    "change",
    [
        {"provider_config": provider("openrouter", True)},
        {"provider_config": provider() | {"model": "another-model"}},
        {"output_limit_policy": None},
    ],
)
def test_focal_and_output_policy_fail_closed(change):
    with pytest.raises(ConfigurationError):
        Config(
            **(
                {
                    "model": "gpt-6.1-sol",
                    "provider_config": provider(),
                    "output_limit_policy": "omit_unsupported",
                }
                | change
            )
        )


def test_usage_limit_does_not_retry_or_substitute(clients):
    sink = Sink()
    instance = model.ChatModel(
        Config(
            model="gpt-6.1-sol",
            provider_config=provider(),
            output_limit_policy="omit_unsupported",
        ),
        sink,
    )

    async def exhausted(*args, **kwargs):
        raise ProviderError("subscription_sharing_usage_limit_exceeded")

    clients[0].complete = exhausted
    with pytest.raises(
        model.ModelError, match="subscription_sharing_usage_limit_exceeded"
    ):
        asyncio.run(
            instance.complete([{"role": "user", "content": "fixture"}], "low", 1)
        )
    assert instance.calls == 1 and len(clients) == 1
    assert "model_retry" not in json.dumps(sink.events)


@pytest.mark.parametrize(
    "strategy,planner",
    [
        ("hard", False),
        ("energy", False),
        ("csp", False),
        ("assertion", False),
        ("llm", False),
        ("hard", True),
        ("energy", True),
        ("csp", True),
        ("assertion", True),
    ],
)
@pytest.mark.parametrize(
    "route,focal", [("chatgpt", True), ("chatgpt", False), ("openrouter", False)]
)
@pytest.mark.parametrize("provider_default", [False, True])
def test_actual_factory_startup_and_close_with_local_synthetic_credentials(
    tmp_path, monkeypatch, strategy, planner, route, focal, provider_default
):
    def no_network(*args, **kwargs):
        pytest.fail("Startup and close must be network-free")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    for key in ("OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    workspace = tmp_path / "agent-final"
    workspace.mkdir()
    monkeypatch.setattr(adapter, "WORKSPACE", workspace / "agent-pins/koichi-sol-v4")
    config = provider(route, focal)
    config["service_tier"] = "synthetic-fast"
    if route == "chatgpt":
        store = CredentialStore(workspace / config["credential_source"])
        record = {
            "account": config["account"],
            "client_id": "synthetic-client",
            "subject": "synthetic-subject",
            "host_id": store.host_id(),
            "access_token": "synthetic-access-token",
            "refresh_token": "synthetic-refresh-token",
            "id_token": "synthetic-identity-token",
            "scopes": SCOPES.split(),
            "expires_at": time.time() + 3600,
        }
        with store._directory() as directory:
            store._write(directory, store._name(config["account"]) + ".json", record)
        cap = ModelCapabilities(
            route="chatgpt",
            model=config["model"],
            credential_source=str(store.root),
            account=config["account"],
            verified_at=time.time(),
            expires_at=time.time() + 3600,
            efforts=("low", "high"),
            effort_evidence="synthetic offline fixture",
            service_tiers=("synthetic-fast",),
            service_tier_evidence="synthetic offline tier fixture",
            fast_service_tiers=("synthetic-fast",),
            fast_evidence="synthetic offline Fast fixture",
            registration_fingerprint=registration_fingerprint(store, config["account"]),
        )
        _write_record(store, config["account"], "capabilities", asdict(cap))
    else:
        source = workspace / config["credential_source"]
        source.parent.mkdir(mode=0o700)
        source.write_text(json.dumps({"api_key": "synthetic-router-key"}))
        source.chmod(0o600)
        capabilities = source.with_name("synthetic-router-capabilities.json")
        capabilities.write_text(
            json.dumps(
                asdict(
                    ModelCapabilities(
                        route="openrouter",
                        model=config["model"],
                        credential_source=str(source),
                        account=None,
                        verified_at=time.time(),
                        expires_at=time.time() + 3600,
                        efforts=("low", "high"),
                        effort_evidence="synthetic offline effort fixture",
                        service_tiers=("synthetic-fast",),
                        service_tier_evidence="synthetic offline tier fixture",
                        fast_service_tiers=("synthetic-fast",),
                        fast_evidence="synthetic offline Fast fixture",
                    )
                )
            )
        )
        capabilities.chmod(0o600)
        config["capabilities_file"] = str(capabilities.relative_to(workspace))
    if provider_default:
        config["require_fast"] = False
        config.pop("service_tier")
        if route == "chatgpt":
            record = asdict(cap)
            record.update(
                efforts=[],
                effort_evidence=None,
                service_tiers=[],
                service_tier_evidence=None,
                fast_service_tiers=[],
                fast_evidence=None,
            )
            _write_record(store, config["account"], "capabilities", record)
    sink = Sink()
    instance = adapter.KoichiAdapter(sink)

    async def run():
        try:
            metadata = await instance.start(
                {
                    "agent": "final",
                    "workspace": str(adapter.WORKSPACE),
                    "record_workspace": str(workspace),
                },
                {
                    "strategy": strategy,
                    "planner_enabled": planner,
                    "model": config["model"],
                    "provider_config": config,
                    "use_provider_default_effort": provider_default,
                    "output_limit_policy": "omit_unsupported"
                    if route == "chatgpt"
                    else "enforce",
                },
            )
            assert isinstance(instance.model._client, InferenceClient)
            assert instance.model._client.config.require_fast is (not provider_default)
            assert instance.model._client.config.service_tier == (
                None if provider_default else "synthetic-fast"
            )
            assert instance.model.calls == 0
            assert instance.agent.config.strategy == strategy
            assert instance.agent.config.planner_enabled is planner
            assert metadata["configuration"]["provider"] == route
            assert metadata["configuration"]["model"] == config["model"]
            assert metadata["configuration"]["focal"] is focal
            assert metadata["configuration"]["copy"] == "koichi-sol-v4"
            assert metadata["configuration"]["require_fast"] is (not provider_default)
            assert metadata["configuration"]["requested_service_tier"] == (
                None if provider_default else "synthetic-fast"
            )
            assert metadata["configuration"]["source_workspace"] == str(
                adapter.WORKSPACE
            )
            assert instance.agent.config.workspace == adapter.WORKSPACE
            assert instance.model._client.config.credential_source.startswith(
                str(workspace / ".private")
            )
            secret = (
                "synthetic-refresh-token"
                if route == "chatgpt"
                else "synthetic-router-key"
            )
            assert (
                sink.redactor.clean(secret)
                == instance.model.redactor.clean(secret)
                == "[REDACTED]"
            )
        finally:
            await instance.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "route,focal", [("chatgpt", True), ("chatgpt", False), ("openrouter", False)]
)
def test_actual_factory_rejects_nonboolean_fast(tmp_path, route, focal):
    with pytest.raises(ProviderError, match="^invalid_require_fast_setting$"):
        build_client(provider(route, focal) | {"require_fast": "false"}, tmp_path)


@pytest.mark.parametrize(
    "tier,code",
    [
        (None, "fast_unverified"),
        ("standard", "fast_unverified"),
        ("ultrafast", "fast_unverified"),
        ("synthetic-fast", "capabilities_unverified"),
    ],
)
def test_actual_factory_rejects_fast_without_verified_evidence(
    tmp_path, monkeypatch, tier, code
):
    def no_network(*args, **kwargs):
        pytest.fail("Invalid Fast configuration must fail before networking")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    workspace = tmp_path / "agent-final"
    source = workspace / ".private/synthetic-source"
    source.parent.mkdir(parents=True, mode=0o700)
    source.write_text(json.dumps({"api_key": "synthetic-router-key"}))
    source.chmod(0o600)
    with pytest.raises(ProviderError, match=f"^{code}$"):
        build_client(provider("openrouter", False) | {"service_tier": tier}, workspace)


@pytest.mark.parametrize("wrong", ["source", "owner"])
def test_controller_context_rejects_foreign_execution_or_owner(
    tmp_path, clients, wrong
):
    instance = adapter.KoichiAdapter(Sink())
    context = {
        "agent": "final",
        "workspace": str(adapter.WORKSPACE),
        "record_workspace": str(tmp_path / "agent-final"),
    }
    context["workspace" if wrong == "source" else "record_workspace"] = str(
        tmp_path / "foreign" / "agent-final"
    )
    with pytest.raises(ConfigurationError, match="workspace"):
        asyncio.run(
            instance.start(
                context,
                {
                    "strategy": "hard",
                    "model": "gpt-6.1-sol",
                    "provider_config": provider(),
                    "output_limit_policy": "omit_unsupported",
                },
            )
        )
    assert clients == [] and instance.agent is None
