"""Authenticate and check authorized assignments without claiming or playing seats."""

import asyncio
import json
from pathlib import Path

from werewolf_runtime.audit import Redactor
from werewolf_runtime.config import official_key
from werewolf_runtime.platform import Control, PlatformError


async def main():
    owner = Path(__file__).resolve().parent / "agent-final"
    control = Control(official_key(owner), Redactor([]))
    try:
        async with asyncio.timeout(30):
            await control.authenticate()
            response = await control.request("GET", "/tournament/agent/assignments")
        assignments = response.get("assignments")
        if not isinstance(assignments, list):
            raise PlatformError("invalid_assignment_response")
        print(
            json.dumps(
                {
                    "authenticated": True,
                    "assignments_reachable": True,
                    "assignment_count": len(assignments),
                    "seats_claimed": 0,
                }
            )
        )
    finally:
        await control.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except PlatformError as error:
        print(
            json.dumps(
                {"connected": False, "error": error.code, "status": error.status}
            )
        )
        raise SystemExit(1) from None
    except Exception:
        print('{"connected":false,"error":"local_setup_or_connectivity_error"}')
        raise SystemExit(1) from None
