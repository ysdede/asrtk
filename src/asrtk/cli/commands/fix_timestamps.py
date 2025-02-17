"""Command for fixing timestamp formats in VTT files."""
from pathlib import Path
import rich_click as click
from rich.console import Console
import re
from typing import List, Tuple

from asrtk.core.vtt import is_timestamp_line, split_vtt_into_chunks, combine_vtt_chunks

console = Console()

def fix_timestamp(line: str) -> Tuple[str, bool]:
    """Fix timestamp format by replacing commas with periods.

    Returns:
        Tuple of (fixed line, whether line was modified)
    """
    if ',' not in line:
        return line, False

    # Only fix timestamps (00:00:00,000 --> 00:00:00,000)
    pattern = r'(\d{2}:\d{2}:\d{2}),(\d{3})'
    fixed = re.sub(pattern, r'\1.\2', line)
    return fixed, fixed != line

def process_vtt_file(vtt_file: Path) -> Tuple[bool, List[Tuple[str, str]], List[str]]:
    """Process a VTT file and fix timestamp formats.

    Returns:
        Tuple of (whether file was modified, list of changes, new content)
    """
    content = vtt_file.read_text(encoding='utf-8')
    new_content = []
    changes = []
    file_modified = False

    for line in content.split('\n'):
        if is_timestamp_line(line):
            fixed_line, was_modified = fix_timestamp(line)
            if was_modified:
                file_modified = True
                changes.append((line, fixed_line))
                new_content.append(fixed_line)
            else:
                new_content.append(line)
        else:
            new_content.append(line)

    return file_modified, changes, new_content

@click.command('fix-timestamps')
@click.argument('input_dir', type=click.Path(exists=True))
@click.option('--output-dir', '-o', type=click.Path(), help="Output directory (default: input_dir + '_fixed')")
@click.option('--dry-run', is_flag=True, help="Show what would be changed without making changes")
@click.option('--verbose', '-v', is_flag=True, help="Show detailed changes")
def fix_timestamps(input_dir: str, output_dir: str | None, dry_run: bool, verbose: bool) -> None:
    """Fix timestamp formats in VTT files.

    Scans VTT files for timestamps using comma as decimal separator (incorrect)
    and replaces them with periods (correct).

    Example incorrect: 00:00:00,000 --> 00:00:00,000
    Example correct:  00:00:00.000 --> 00:00:00.000

    Examples:
        # Check for incorrect timestamps
        asrtk fix-timestamps ./subtitles --dry-run

        # Fix timestamps
        asrtk fix-timestamps ./subtitles -o ./fixed_subtitles
    """
    input_path = Path(input_dir)
    if not output_dir:
        output_dir = str(input_path) + '_fixed'
    output_path = Path(output_dir)

    # Find all VTT files
    vtt_files = list(input_path.rglob("*.vtt"))
    if not vtt_files:
        console.print("No VTT files found in input directory")
        return

    console.print(f"Found {len(vtt_files)} VTT files")

    if not dry_run:
        output_path.mkdir(parents=True, exist_ok=True)

    # Process files
    files_modified = 0
    total_changes = 0

    for vtt_file in vtt_files:
        try:
            file_modified, changes, new_content = process_vtt_file(vtt_file)

            if file_modified:
                files_modified += 1
                total_changes += len(changes)

                if verbose or dry_run:
                    console.print(f"\nIn [blue]{vtt_file}[/blue]:")
                    console.print(f"  {len(changes)} timestamps to fix")
                    if changes and (verbose or dry_run):
                        console.print("  Changes:")
                        for old, new in changes[:3]:  # Show up to 3 examples
                            console.print(f"    [red]{old}[/red] → [green]{new}[/green]")
                        if len(changes) > 3:
                            console.print(f"    ... and {len(changes) - 3} more")

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
        console.print(f"Would fix {total_changes} timestamps")
    else:
        console.print(f"\nProcessed {len(vtt_files)} files")
        console.print(f"Modified {files_modified} files")
        console.print(f"Fixed {total_changes} timestamps")
        console.print(f"Results saved to {output_path}")
