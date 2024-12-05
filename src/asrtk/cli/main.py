"""Main CLI entry point for asrtk."""
import rich_click as click
from .. import __version__

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

# Import and register commands
from .commands.wordset import create_wordset
from .commands.patch import apply_patch
from .commands.download import download_playlist, download_channel
from .commands.fix import fix
from .commands.find import find_words, find_arabic, find_patterns, find_brackets
from .commands.split import split
from .commands.merge import merge_lines
from .commands.remove import remove_lines

# Register commands
cli.add_command(create_wordset)
cli.add_command(apply_patch)
cli.add_command(download_playlist)
cli.add_command(download_channel)
cli.add_command(fix)
cli.add_command(find_words)
cli.add_command(find_arabic)
cli.add_command(find_patterns)
cli.add_command(find_brackets)
cli.add_command(split)
cli.add_command(merge_lines)
cli.add_command(remove_lines)

if __name__ == "__main__":
    cli()
