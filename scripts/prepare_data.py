"""Convenience entry point for schedulers that prefer a script path."""

from remit.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["data", "prepare"]))
