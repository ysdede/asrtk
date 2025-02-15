"""Main CLI entry point for asrtk."""
import importlib
from typing import Callable, Dict, Tuple
import rich_click as click

# Version is the only thing we need from the package
# from .. import __version__
version = "0.1.3"

context_settings = {"help_option_names": ["-h", "--help"]}
help_config = click.RichHelpConfiguration(
    width=88,
    show_arguments=True,
    use_rich_markup=True,
)

# Dictionary mapping command names to their import paths and functions
COMMAND_REGISTRY: Dict[str, Tuple[str, str, str]] = {
    # Format: 'command-name': (module_path, function_name, help_text)
    "download-playlist": (".commands.download", "download_playlist", "Download videos from a YouTube playlist with subtitles"),
    "download-channel": (".commands.download", "download_channel", "Download videos from a YouTube channel with subtitles"),
    "create-wordset": (".commands.wordset", "create_wordset", "Create a wordset from subtitle files"),
    "apply-patch": (".commands.patch", "apply_patch", "Apply a patch file to subtitles"),
    "fix": (".commands.fix", "fix", "Fix common issues in subtitle files"),
    "find-words": (".commands.find", "find_words", "Find specific words in subtitle files"),
    "find-arabic": (".commands.find", "find_arabic", "Find Arabic text in subtitle files"),
    "find-patterns": (".commands.find", "find_patterns", "Find regex patterns in subtitle files"),
    "find-brackets": (".commands.find", "find_brackets", "Find bracketed text in subtitle files"),
    "split": (".commands.split", "split", "Split subtitle files"),
    "merge-lines": (".commands.merge", "merge_lines", "Merge consecutive subtitle lines"),
    "remove-lines": (".commands.remove", "remove_lines", "Remove specific lines from subtitle files"),
    "count-numbers": (".commands.numbers", "count_numbers", "Count numbers in subtitle files"),
    "find-abbreviations": (".commands.abbreviations", "find_abbreviations", "Find abbreviations in subtitle files"),
    "fix-timestamps": (".commands.fix_timestamps", "fix_timestamps", "Fix subtitle timestamps"),
    "convert": (".commands.convert", "convert", "Convert subtitle files between formats"),
    "probe-audio": (".commands.convert", "probe_audio", "Probe audio files for metadata"),
    "probe-mp3": (".commands.convert", "probe_mp3", "Probe MP3 files for metadata"),
    "chunk": (".commands.chunk", "chunk", "Split audio and subtitle files into chunks"),
    "duplicates": (".commands.duplicates", "duplicates", "Find duplicate subtitle lines"),
    "trim": (".commands.trim", "trim", "Trim audio and subtitle files"),
    "create-charset": (".commands.charset", "create_charset", "Create a character set from subtitle files"),
}

class LazyCommand(click.Command):
    """A Click Command that loads its implementation only when invoked."""

    def __init__(
        self,
        name: str,
        module_path: str,
        function_name: str,
        help_text: str,
        **kwargs
    ):
        super().__init__(name=name, help=help_text, **kwargs)
        self.module_path = module_path
        self.function_name = function_name
        self._loaded_command: Callable | None = None

    def invoke(self, ctx: click.Context) -> None:
        """Load and invoke the actual command implementation."""
        if self._loaded_command is None:
            try:
                module = importlib.import_module(self.module_path, package="asrtk.cli")
                self._loaded_command = getattr(module, self.function_name)
                # Copy the signature and help from the loaded command
                self.params = self._loaded_command.params
                self.callback = self._loaded_command
            except ImportError as e:
                raise click.ClickException(
                    f"Failed to load command {self.name}: {str(e)}"
                )
        return super().invoke(ctx)

@click.group(context_settings=context_settings)
@click.rich_config(help_config=help_config)
@click.version_option(version, "-v", "--version")
def cli() -> None:
    """An open-source Python toolkit designed to streamline the development and enhancement of ASR systems."""

# Register lazy-loaded commands
for cmd_name, (module_path, func_name, help_text) in COMMAND_REGISTRY.items():
    cli.add_command(LazyCommand(
        name=cmd_name,
        module_path=module_path,
        function_name=func_name,
        help_text=help_text
    ))

def main():
    """Entry point for the CLI."""
    cli()
