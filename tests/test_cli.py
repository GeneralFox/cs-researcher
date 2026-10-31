import json
import subprocess
import sys

import pytest

from cs_researcher.cli import COMMANDS, build_parser, main


@pytest.mark.parametrize("command", [c.name for c in COMMANDS])
def test_every_command_has_help(command):
    result = subprocess.run(
        [sys.executable, "-m", "cs_researcher", command, "--help"], capture_output=True, text=True, check=True
    )
    assert result.stdout.startswith("usage: cs-researcher")


def test_dashboard_does_not_import_crewai():
    code = "import sys, cs_researcher.cli; assert 'crewai' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_config_command_prints_json(capsys):
    assert main(["config"]) == 0
    assert json.loads(capsys.readouterr().out)["coordinator"]["name"] == "research_coordinator"


def test_bad_config_exits_2(tmp_path, capsys):
    path = tmp_path / "bad.json"
    path.write_text("{}")
    assert main(["--config", str(path), "config"]) == 2
    assert "invalid configuration" in capsys.readouterr().err


def test_a_command_is_required():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])
