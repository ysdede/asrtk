# CLI Reference

This page lists the `--help` for `asrtk` and all its commands.

## asrtk

Running `asrtk --help` or `python -m asrtk --help` shows a list of all of the available options and commands:

<!-- [[[cog
import cog
from asrtk import cli
from click.testing import CliRunner
result = CliRunner().invoke(cli.cli, ["--help"], terminal_width=88)
help = result.output.replace("Usage: cli", "Usage: asrtk")
cog.outl(f"\n```sh\nasrtk --help\n{help.rstrip()}\n```\n")
for command in cli.cli.commands.keys():
    result = CliRunner().invoke(cli.cli, [command, "--help"], terminal_width=88)
    help = result.output.replace("Usage: cli ", "Usage: asrtk ")
    cog.outl(f"## asrtk {command}\n\n```sh\nasrtk {command} --help\n{help.rstrip()}\n```\n")
]]] -->
<!-- [[[end]]] -->
