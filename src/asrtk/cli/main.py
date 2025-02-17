"""Main CLI entry point for asrtk."""
import rich_click as click
from functools import partial, update_wrapper
from importlib import import_module

# Version
version = "0.1.3"

# Click configuration
context_settings = {"help_option_names": ["-h", "--help"]}
help_config = click.RichHelpConfiguration(
    width=88,
    show_arguments=True,
    use_rich_markup=True,
)

def load_command(module_path: str, command_name: str):
    """Load a command from a module."""
    print(f"[asrtk.cli.main] Importing {command_name} from {module_path}")
    module = import_module(module_path)
    return getattr(module, command_name)

@click.group(context_settings=context_settings)
@click.rich_config(help_config=help_config)
@click.version_option(version=version)
def cli():
    """An open-source Python toolkit designed to streamline the development and enhancement of ASR systems."""
    pass

# Command definitions with their module paths
commands = [
    ("download-playlist", "asrtk.cli.commands.download", "download_playlist"),
    ("download-channel", "asrtk.cli.commands.download", "download_channel"),
    ("download-channel-wosub", "asrtk.cli.commands.download", "download_channel_wosub"),
    ("create-wordset", "asrtk.cli.commands.wordset", "create_wordset"),
    ("apply-patch", "asrtk.cli.commands.patch", "apply_patch"),
    ("fix", "asrtk.cli.commands.fix", "fix"),
    ("find-words", "asrtk.cli.commands.find", "find_words"),
    ("find-arabic", "asrtk.cli.commands.find", "find_arabic"),
    ("find-patterns", "asrtk.cli.commands.find", "find_patterns"),
    ("find-brackets", "asrtk.cli.commands.find", "find_brackets"),
    ("split", "asrtk.cli.commands.split", "split"),
    ("merge-lines", "asrtk.cli.commands.merge", "merge_lines"),
    ("remove-lines", "asrtk.cli.commands.remove", "remove_lines"),
    ("count-numbers", "asrtk.cli.commands.numbers", "count_numbers"),
    ("find-abbreviations", "asrtk.cli.commands.abbreviations", "find_abbreviations"),
    ("fix-timestamps", "asrtk.cli.commands.fix_timestamps", "fix_timestamps"),
    ("convert", "asrtk.cli.commands.convert", "convert"),
    ("probe-audio", "asrtk.cli.commands.convert", "probe_audio"),
    ("probe-mp3", "asrtk.cli.commands.convert", "probe_mp3"),
    ("chunk", "asrtk.cli.commands.chunk", "chunk"),
    ("duplicates", "asrtk.cli.commands.duplicates", "duplicates"),
]

# Register commands
for cmd_name, module_path, func_name in commands:
    # Load the command to get its metadata
    command = load_command(module_path, func_name)
    # Register the command with Click
    cli.add_command(command, cmd_name)

def main():
    """Entry point for the CLI."""
    cli()
