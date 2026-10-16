"""An isolated adapter subprocess, with no inherited platform/model credentials."""

import asyncio
import json
import os
import signal

from .config import AgentSpec
from .protocol import MAX_LINE_BYTES


class AdapterError(RuntimeError):
    pass


class Worker:
    def __init__(self, spec: AgentSpec, record) -> None:
        self.spec, self.record = spec, record
        self.process = None
        self.counter = 0

    async def launch(self) -> None:
        environment = {
            key: os.environ[key]
            for key in (
                "PATH",
                "HOME",
                "LANG",
                "LC_ALL",
                "SSL_CERT_FILE",
                "SSL_CERT_DIR",
            )
            if key in os.environ
        }
        self.process = await asyncio.create_subprocess_exec(
            *self.spec.command,
            cwd=self.spec.workspace,
            env=environment,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            # Untrusted stderr can contain secrets or hidden state. Never echo it.
            stderr=asyncio.subprocess.DEVNULL,
            limit=MAX_LINE_BYTES,
            start_new_session=True,
        )

    async def request(self, method: str, params: dict, timeout: float = 15) -> object:
        self.counter += 1
        request_id = self.counter

        async def exchange():
            if self.process is None or self.process.returncode is not None:
                raise AdapterError("Adapter is not running.")
            payload = (
                json.dumps(
                    {"id": request_id, "method": method, "params": params},
                    ensure_ascii=True,
                    allow_nan=False,
                ).encode()
                + b"\n"
            )
            if len(payload) > MAX_LINE_BYTES:
                raise AdapterError("Adapter request exceeds the protocol limit.")
            self.process.stdin.write(payload)
            await self.process.stdin.drain()
            while True:
                line = await self.process.stdout.readline()
                if not line or len(line) > MAX_LINE_BYTES:
                    raise AdapterError("Adapter closed or exceeded the protocol limit.")
                response = json.loads(line)
                if not isinstance(response, dict):
                    raise AdapterError("Invalid adapter response.")
                if response.get("type") == "event":
                    if set(response) != {"type", "event", "data"} or not isinstance(
                        response["event"], str
                    ):
                        raise AdapterError("Invalid adapter event.")
                    self.record(response["event"], response["data"])
                    continue
                if (
                    type(response.get("id")) is not int
                    or response["id"] != request_id
                    or set(response) != {"id", "result"}
                ):
                    raise AdapterError(
                        "Adapter failed or returned a mismatched request ID."
                    )
                return response["result"]

        try:
            return await asyncio.wait_for(exchange(), timeout)
        except OSError, ValueError, TimeoutError:
            raise AdapterError(
                "Adapter request failed or exceeded its deadline."
            ) from None

    async def close(self) -> None:
        if self.process is None:
            return
        if self.process.returncode is None:
            try:
                await self.request("close", {}, 3)
                await asyncio.wait_for(self.process.wait(), 3)
            except AdapterError, TimeoutError, ProcessLookupError:
                pass
        # Children can survive even after the group leader has already exited.
        try:
            os.killpg(self.process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await self.process.wait()
