"""Seat-local policy adapter, shared by standalone and Testing execution."""

import asyncio
from dataclasses import fields
from pathlib import Path

from final_providers.factory import resolved_metadata
from werewolf_runtime.protocol import EventSink, serve

from .agent import WerewolfAgent
from .config import WORKSPACE, Config, ConfigurationError
from .model import ChatModel


class KoichiAdapter:
    def __init__(self, sink: EventSink) -> None:
        self.sink = sink
        self.model: ChatModel | None = None
        self.agent: WerewolfAgent | None = None
        self._closing = None

    async def start(self, context: dict, config: dict) -> dict:
        if self.model is not None or self._closing is not None:
            raise ConfigurationError("Adapter already started.")
        workspace = context.get("workspace")
        source = WORKSPACE.absolute()
        owner = source.parents[1]
        record_workspace = context.get("record_workspace", str(owner))
        if (
            context.get("agent") != "final"
            or not isinstance(workspace, str)
            or not Path(workspace).is_absolute()
            or source != source.resolve()
            or source.name != "koichi-sol-v4"
            or source.parent.name != "agent-pins"
            or owner.name != "agent-final"
            or Path(workspace) != Path(workspace).resolve()
            or Path(workspace) not in (source, owner)
            or not isinstance(record_workspace, str)
            or Path(record_workspace) != owner
        ):
            raise ConfigurationError(
                "Runner context must identify the final record workspace."
            )
        allowed = {entry.name for entry in fields(Config)} - {
            "workspace",
            "model_key",
            "official_key",
        }
        if set(config) - allowed:
            raise ConfigurationError(
                "Runner configuration contains unsupported fields."
            )
        if (
            not isinstance(config.get("strategy"), str)
            or not config["strategy"].strip()
        ):
            raise ConfigurationError(
                "Runner configuration requires an explicit strategy."
            )
        configuration = Config.from_env(**config, workspace=source, model_key="")
        configuration.require_model()
        self.model = ChatModel(configuration, self.sink)
        metadata = {
            "strategy": configuration.strategy,
            "configuration": {
                **configuration.runtime_metadata(),
                **resolved_metadata(self.model._client),
                "copy": "koichi-sol-v4",
                "record_workspace": "agent-final",
                "source_workspace": str(source),
            },
            "decision_seconds": configuration.deep_seconds,
            "max_chats": configuration.max_chats,
        }
        self.agent = WerewolfAgent(configuration, self.model, self.sink)
        return metadata

    async def decide(
        self, state: dict, messages: list[dict], kind: str, seconds: float
    ) -> dict:
        if self.agent is None:
            raise ConfigurationError("Adapter is not started.")
        if kind not in ("action", "message"):
            raise ValueError("Unknown decision kind.")
        answer = await self.agent.choose(state, messages, kind, seconds)
        decision = answer["decision"]
        result = (
            {"action_id": decision["action_id"]}
            if kind == "action"
            else {
                "content": decision["content"].strip(),
                "recipients": decision["recipients"],
            }
        )
        self.sink.record("adapter_decision", {"kind": kind, "answer": answer})
        return result

    async def close(self) -> None:
        if self._closing is None:

            async def finish():
                try:
                    if self.agent is not None:
                        await self.agent.close()
                finally:
                    try:
                        if self.model is not None:
                            await self.model.close()
                    finally:
                        self.agent = self.model = None

            self._closing = asyncio.create_task(finish())
        await asyncio.shield(self._closing)


if __name__ == "__main__":
    raise SystemExit(serve(KoichiAdapter))
