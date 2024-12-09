"""Command for removing lines containing specific patterns."""
from pathlib import Path
import rich_click as click
from rich.console import Console
from typing import List, Tuple

console = Console()

def process_vtt_file(vtt_file: Path, pattern: str, dry_run: bool = False) -> Tuple[bool, List[str], List[str]]:
    """Process a single VTT file.

    Args:
        vtt_file: Path to VTT file
        pattern: Text pattern to match for line removal
        dry_run: Whether this is a dry run

    Returns:
        Tuple of (whether file was modified, list of removed lines, new content)
    """
    content = vtt_file.read_text(encoding='utf-8')
    new_content = []
    removed_lines = []
    file_modified = False

    for line in content.split('\n'):
        # Skip VTT header, timestamps, line numbers and empty lines
        if not line.strip() or line.startswith('WEBVTT') or '-->' in line or line.strip().isdigit():
            new_content.append(line)
            continue

        # Check if line contains the pattern
        if pattern in line:
            new_content.append(" ")  # Replace with single space
            removed_lines.append(line)
            file_modified = True
        else:
            new_content.append(line)

    return file_modified, removed_lines, new_content

@click.command('remove-lines')
@click.argument('pattern', type=str)
@click.argument('input_dir', type=click.Path(exists=True))
@click.option('--output-dir', '-o', type=click.Path(), help="Output directory (default: input_dir + '_cleaned')")
@click.option('--dry-run', is_flag=True, help="Show what would be removed without making changes")
@click.option('--verbose', '-v', is_flag=True, help="Show removed lines")
def remove_lines(pattern: str, input_dir: str, output_dir: str | None, dry_run: bool, verbose: bool) -> None:
    """Remove lines containing specific pattern from VTT files.

    This command processes VTT files and replaces any line containing the specified pattern
    with a single space, while preserving VTT structure (headers, timestamps, etc).

    Example:
        # Remove all lines containing "silsil888"
        asrtk remove-lines silsil888 ./subtitles --dry-run
    """
    input_path = Path(input_dir)
    if not output_dir:
        output_dir = str(input_path) + '_cleaned'
    output_path = Path(output_dir)

    # Find all VTT files
    vtt_files = list(input_path.rglob("*.vtt"))
    if not vtt_files:
        console.print("No VTT files found in input directory")
        return

    console.print(f"Found {len(vtt_files)} VTT files")
    console.print(f"Looking for pattern: [yellow]{pattern}[/yellow]")

    if not dry_run:
        output_path.mkdir(parents=True, exist_ok=True)

    # Process files
    files_modified = 0
    total_lines_removed = 0

    for vtt_file in vtt_files:
        try:
            file_modified, removed_lines, new_content = process_vtt_file(vtt_file, pattern, dry_run)

            if file_modified:
                files_modified += 1
                total_lines_removed += len(removed_lines)

                if verbose or dry_run:
                    console.print(f"\nIn [blue]{vtt_file}[/blue]:")
                    console.print(f"  {len(removed_lines)} lines removed")
                    if removed_lines and (verbose or dry_run):
                        console.print("  Removed lines:")
                        for line in removed_lines[:3]:  # Show up to 3 examples
                            console.print(f"    [red]{line}[/red]")
                        if len(removed_lines) > 3:
                            console.print(f"    ... and {len(removed_lines) - 3} more")

                if not dry_run:
                    # Write the modified content
                    rel_path = vtt_file.relative_to(input_path)
                    out_file = output_path / rel_path
                    out_file.parent.mkdir(parents=True, exist_ok=True)
                    out_file.write_text('\n'.join(new_content), encoding='utf-8')

        except Exception as e:
            console.print(f"\n[red]Error processing {vtt_file}: {e}[/red]")
            continue

    if dry_run:
        console.print(f"\nDry run complete. Would modify {files_modified} files")
        console.print(f"Would remove {total_lines_removed} lines")
    else:
        console.print(f"\nProcessed {len(vtt_files)} files")
        console.print(f"Modified {files_modified} files")
        console.print(f"Removed {total_lines_removed} lines")
        console.print(f"Results saved to {output_path}")
