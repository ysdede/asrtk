"""Command for merging multi-line subtitles into single lines."""
from pathlib import Path
import rich_click as click
from rich.console import Console
from typing import List, Tuple

# VTT header elements we want to preserve
VTT_HEADERS = {'WEBVTT', 'Kind: captions', 'Language:'}

console = Console()

def is_timestamp_line(line: str) -> bool:
    """Check if line contains timestamp."""
    return '-->' in line

def is_header_line(line: str) -> bool:
    """Check if line is a VTT header line."""
    return any(line.startswith(header) for header in VTT_HEADERS)

def process_vtt_file(vtt_file: Path, dry_run: bool = False) -> Tuple[bool, List[str]]:
    """Process a single VTT file.

    Args:
        vtt_file: Path to VTT file
        dry_run: Whether this is a dry run

    Returns:
        Tuple of (whether file was modified, list of merged lines)
    """
    content = vtt_file.read_text(encoding='utf-8')
    lines = content.split('\n')

    new_content = []
    current_subtitle = []
    file_modified = False

    for line in lines:
        # Preserve empty lines
        if not line.strip():
            if current_subtitle:
                # Join accumulated subtitle lines
                new_content.append(' '.join(current_subtitle))
                current_subtitle = []
            new_content.append(line)
            continue

        # Preserve header lines
        if is_header_line(line):
            if current_subtitle:
                new_content.append(' '.join(current_subtitle))
                current_subtitle = []
            new_content.append(line)
            continue

        # Preserve timestamp lines
        if is_timestamp_line(line):
            if current_subtitle:
                new_content.append(' '.join(current_subtitle))
                current_subtitle = []
            new_content.append(line)
            continue

        # Accumulate subtitle lines
        current_subtitle.append(line.strip())
        file_modified = True

    # Handle any remaining subtitle lines
    if current_subtitle:
        new_content.append(' '.join(current_subtitle))

    return file_modified, new_content

@click.command('merge-lines')
@click.argument('input_dir', type=click.Path(exists=True))
@click.option('--output-dir', '-o', type=click.Path(), help="Output directory (default: input_dir + '_merged')")
@click.option('--dry-run', is_flag=True, help="Show what would be merged without making changes")
@click.option('--verbose', '-v', is_flag=True, help="Show detailed changes")
def merge_lines(input_dir: str, output_dir: str | None, dry_run: bool, verbose: bool) -> None:
    """Merge multi-line subtitles into single lines while preserving VTT structure.

    This command processes VTT files and merges subtitle text that spans multiple lines
    into single lines, while preserving timestamps and VTT header information.

    Example input:
        00:00:04.280 --> 00:00:05.960
        Çünkü bir daha öyle
        bir şey yaşamadık.

    Example output:
        00:00:04.280 --> 00:00:05.960
        Çünkü bir daha öyle bir şey yaşamadık.
    """
    input_path = Path(input_dir)
    if not output_dir:
        output_dir = str(input_path) + '_merged'
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

    for vtt_file in vtt_files:
        try:
            file_modified, new_content = process_vtt_file(vtt_file, dry_run)

            if file_modified:
                files_modified += 1

                if verbose or dry_run:
                    console.print(f"\nIn [blue]{vtt_file}[/blue]:")

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
    else:
        console.print(f"\nProcessed {len(vtt_files)} files")
        console.print(f"Modified {files_modified} files")
        console.print(f"Results saved to {output_path}")
