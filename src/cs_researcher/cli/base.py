"""Base class of the cs-researcher subcommands."""

import argparse
from abc import ABC, abstractmethod
from typing import ClassVar

from ..config import AppConfig


class Command(ABC):
    """A subcommand: subclass it and add an instance to cli.COMMANDS."""

    name: ClassVar[str]
    help: ClassVar[str]

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:  # noqa: B027 (optional hook)
        """Add the command's own options to its parser (none by default)."""

    @abstractmethod
    def run(self, args: argparse.Namespace, config: AppConfig) -> int:
        """Run the command and return the exit code."""
