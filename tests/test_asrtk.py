from click.testing import CliRunner

import pytest
from asrtk.cli import cli


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_version(runner: CliRunner) -> None:
    with runner.isolated_filesystem():
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert result.output.startswith("cli, version ")
