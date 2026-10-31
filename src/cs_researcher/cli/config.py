import argparse

from ..config import AppConfig
from .base import Command


class ConfigCommand(Command):
    name = "config"
    help = "print the configuration in use, as JSON (a starting point for --config)"

    def run(self, args: argparse.Namespace, config: AppConfig) -> int:
        print(config.to_json())
        return 0
