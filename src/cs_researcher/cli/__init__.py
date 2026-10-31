"""Command line: `cs-researcher [--config FILE] <command> ...`."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from .. import __version__
from ..config import AppConfig, ConfigError
from .ask import AskCommand
from .base import Command
from .config import ConfigCommand
from .dashboard import DashboardCommand

COMMANDS: list[Command] = [AskCommand(), DashboardCommand(), ConfigCommand()]


def build_parser(commands: Sequence[Command] = COMMANDS) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cs-researcher",
        description="Multi-agent cybersecurity research on local Ollama models.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--config",
        type=Path,
        metavar="FILE",
        help="JSON configuration file (default: the one shipped with the package)",
    )
    subparsers = parser.add_subparsers(title="commands", metavar="<command>", required=True)
    for command in commands:
        sub = subparsers.add_parser(command.name, help=command.help, description=command.help)
        command.add_arguments(sub)
        sub.set_defaults(command=command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = AppConfig.load(args.config)
    except ConfigError as e:
        print(f"cs-researcher: {e}", file=sys.stderr)
        return 2
    return args.command.run(args, config)
