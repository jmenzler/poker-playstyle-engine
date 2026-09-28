"""src/cli — poker-engine console entrypoints (Phase 4+).

Subcommands are registered under a single argparse dispatcher in main.py,
which is wired as a console_scripts entry in pyproject.toml.

Each subcommand module exposes a ``run(args) -> int`` function that receives
the parsed ``argparse.Namespace`` from the dispatcher.
"""
