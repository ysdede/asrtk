"""Command for applying character replacements to VTT files."""
from pathlib import Path
import rich_click as click
import json
from typing import Dict, List, Tuple
from rich.console import Console

from ...utils.console import (
    print_file_header,
    print_replacement_example
)
from ...utils.file import backup_file

console = Console()


def print_patch_diff(old_line: str, new_line: str, word: str, replacement: str) -> None:
    """Print a diff of the patch applied to a line."""
    # Print difference between old and new line, as print old line and highlight the difference, Do not use click's console
    # In some cases, replacement may be "", empty string, so we need to handle that
    old_line_parts = old_line.split(word)
    if replacement:
        new_line_parts = new_line.split(replacement)
        print(f"'{old_line_parts[0]}{click.style(word, fg='red')}{old_line_parts[1]}' → '{new_line_parts[0]}{click.style(replacement, fg='green')}{new_line_parts[1]}'")
    else:# print old line in color as we do print(f"{old_line_parts[0]}{click.style(word, fg='red')}{old_line_parts[1]} → {new_line_parts[0]}{click.style(replacement, fg='green')}{new_line_parts[1]}")  then print the new line
        print(f"'{old_line_parts[0]}{click.style(word, fg='red')}{old_line_parts[1]}' → '{new_line}'")



def load_replacements(patch_file: Path) -> Dict[str, str]:
    """Load replacements from patch file.

    Args:
        patch_file: Path to patch file

    Returns:
        Dictionary of replacements
    """
    with patch_file.open('r', encoding='utf-8') as f:
        patch_data = json.load(f)

    replacements = {}
    for item in patch_data:
        # Handle both char and sequence cases
        key = item.get('char') or item.get('sequence')
        replacement = item.get('replacement')  # Don't default to empty string

        if key and replacement is not None:  # Only add if key exists and replacement is defined
            replacements[key] = replacement

    return replacements


def process_vtt_file(vtt_file: Path,
                     replacements: Dict[str, str],
                     verbose: bool = False,
                     dry_run: bool = False) -> Tuple[int, List[Tuple[str, str, str, str]]]:
    """Process a single VTT file with replacements.

    Args:
        vtt_file: Path to VTT file
        replacements: Dictionary of replacements
        verbose: Whether to collect verbose information
        dry_run: Whether this is a dry run

    Returns:
        Tuple of (number of replacements, list of example changes)
    """
    content = vtt_file.read_text(encoding='utf-8')
    new_content = []
    file_chars_replaced = 0
    changes = []

    # Sort replacements by length (longest first) to avoid partial matches
    sorted_replacements = sorted(
        replacements.items(), key=lambda x: len(x[0]), reverse=True)

    # Process line by line
    for line in content.split('\n'):
        # Skip VTT header, timestamps, line numbers and empty lines
        if not line.strip() or line.strip() == 'WEBVTT' or '-->' in line or line.strip().isdigit():
            new_content.append(line)
            continue

        new_line = line
        original_line = line

        # Apply replacements
        for old_text, new_text in sorted_replacements:
            if old_text in new_line:
                count = new_line.count(old_text)
                new_line = new_line.replace(old_text, new_text)
                file_chars_replaced += count

                # Store example if line was changed and we haven't stored too many
                if (verbose or dry_run) and original_line != new_line and len(changes) < 5:
                    changes.append(
                        (original_line, new_line, old_text, new_text))

        new_content.append(new_line)

    return file_chars_replaced, changes, '\n'.join(new_content)


@click.command()
@click.argument("patch_file", type=click.Path(exists=True), required=False)
@click.argument("input_dir", type=click.Path(exists=True), required=False)
@click.option("--output-dir", "-o", type=click.Path(), help="Output directory (default: input_dir + '_patched')")
@click.option("--dry-run", is_flag=True, help="Show what would be replaced without making changes")
@click.option("--show-replacements", "-s", is_flag=True, help="Show detailed replacement plan")
@click.option("--create-template", is_flag=True, help="Create a template patch file and exit")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed changes in each file")
def apply_patch(patch_file: str | None,
                input_dir: str | None,
                output_dir: str | None,
                dry_run: bool,
                show_replacements: bool,
                create_template: bool,
                verbose: bool) -> None:
    """Apply character replacements from a patch file to VTT files.

    Uses a simple JSON patch file containing character replacements.
    The patch file should contain an array of objects with 'char' and 'replacement' fields.
    Supports both single and multi-character replacements in both directions.

    Example patch file:
    [
        {
            "char": "ú",
            "replacement": "ue",
            "description": "Replace single char with multiple chars"
        },
        {
            "sequence": "...",
            "replacement": "…",
            "description": "Replace multiple chars with single char"
        }
    ]

    Create a template with: asrtk apply-patch --create-template
    """
    if create_template:
        template = [
            {
                "char": "ú",
                "replacement": "ue",
                "description": "Replace acute u with ue"
            },
            {
                "sequence": "</i>",
                "replacement": "",  # Empty string to remove the sequence
                "description": "Remove italic tags"
            },
            {
                "sequence": "<i>",
                "replacement": None,  # null also removes the sequence
                "description": "Remove italic tags"
            },
            {
                "sequence": " ?",
                "replacement": "?",
                "description": "Remove space before question mark"
            }
        ]
        click.echo(json.dumps(template, ensure_ascii=False, indent=2))
        return

    # Validate required arguments when not creating template
    if not patch_file:
        raise click.UsageError(
            "PATCH_FILE is required when not using --create-template")
    if not input_dir:
        raise click.UsageError(
            "INPUT_DIR is required when not using --create-template")

    patch_path = Path(patch_file)
    input_dir = Path(input_dir)

    if not output_dir:
        output_dir = str(input_dir) + '_patched'
    output_dir = Path(output_dir)

    # Load replacements
    replacements = load_replacements(patch_path)
    print(replacements)

    if not replacements:
        click.echo("No valid replacements found in patch file")
        return

    click.echo(f"Loaded {len(replacements)} replacements")

    # Show detailed replacement plan if requested
    if show_replacements:
        click.echo("\nPlanned replacements:")
        for old, new in replacements.items():
            desc = next((item.get('description', '') for item in json.loads(patch_path.read_text())
                        if item.get('char') == old or item.get('sequence') == old), '')
            click.echo(f"{repr(old)} -> {repr(new)}" +
                       (f" # {desc}" if desc else ""))

        if not click.confirm("\nProceed with these replacements?"):
            click.echo("Operation cancelled")
            return

    # Find all VTT files
    vtt_files = list(input_dir.rglob("*.vtt"))
    if not vtt_files:
        click.echo("No VTT files found in input directory")
        return

    click.echo(f"Found {len(vtt_files)} VTT files")

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Process files
    files_changed = 0
    chars_replaced = 0

    # Don't use progress bar when verbose output is needed
    if verbose or dry_run:
        files_iter = vtt_files
        console.print(f"Processing {len(vtt_files)} files...")
    else:
        files_iter = click.progressbar(vtt_files, label='Processing files')

    try:
        for vtt_file in files_iter:
            # try:
            file_chars_replaced, changes, new_content = process_vtt_file(
                vtt_file, replacements, verbose, dry_run)

            if file_chars_replaced > 0:
                files_changed += 1
                chars_replaced += file_chars_replaced

                # Show changes if requested
                if dry_run or verbose:
                    console.print(f"\nIn [blue]{vtt_file}[/blue]:")
                    console.print(f"  {file_chars_replaced} replacements")
                    if changes:
                        console.print("  Example changes:")
                        for original_line, new_line, old_text, new_text in changes:
                            print_patch_diff(
                                original_line, new_line, old_text, new_text)

                if not dry_run:
                    # Write the modified content
                    rel_path = vtt_file.relative_to(input_dir)
                    out_file = output_dir / rel_path
                    out_file.parent.mkdir(parents=True, exist_ok=True)
                    out_file.write_text(new_content, encoding='utf-8')

            # except Exception as e:
            #     console.print(f"\n[red]Error processing {vtt_file}: {e}[/red]")
            #     continue
    finally:
        if not (verbose or dry_run):
            files_iter.finish()  # Clean up progress bar

    if dry_run:
        click.echo(
            f"\nDry run complete. Would modify {files_changed} files with {chars_replaced} replacements")
    else:
        click.echo(f"\nProcessed {len(vtt_files)} files")
        click.echo(f"Modified {files_changed} files")
        click.echo(f"Made {chars_replaced} character replacements")
        click.echo(f"Results saved to {output_dir}")
