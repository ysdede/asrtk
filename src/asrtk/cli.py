"""Main CLI for asrtk."""
import rich_click as click

from . import __version__

context_settings = {"help_option_names": ["-h", "--help"]}
help_config = click.RichHelpConfiguration(
    width=88,
    show_arguments=True,
    use_rich_markup=True,
)


@click.group(context_settings=context_settings)
@click.rich_config(help_config=help_config)
@click.version_option(__version__, "-v", "--version")
def cli() -> None:
    """An open-source Python toolkit designed to streamline the development and enhancement of ASR systems."""


@cli.command()
@click.argument("input_", metavar="INPUT")
@click.option(
    "-r",
    "--reverse",
    is_flag=True,
    help="Reverse the input.",
)
def repeat(input_: str, *, reverse: bool = False) -> None:
    """Repeat the input."""
    click.echo(input_ if not reverse else input_[::-1])
