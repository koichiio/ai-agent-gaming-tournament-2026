"""Run only the explicitly selected operator command."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
