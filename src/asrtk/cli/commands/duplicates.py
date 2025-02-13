"""Command for finding and managing duplicate files based on YouTube IDs."""
from pathlib import Path
from collections import defaultdict
import re
import shutil
import json
import rich_click as click
from rich.console import Console
from rich.table import Table
from rich.progress import track

def extract_youtube_id(filename: str) -> str:
    """Extract YouTube ID from filename containing [YOUTUBE_ID].

    Args:
        filename: Filename potentially containing [YOUTUBE_ID]

    Returns:
        YouTube ID if found, otherwise original filename without extension
    """
    # Look for [SOMETHING] pattern
    matches = re.findall(r'\[(.*?)\]', filename)
    if matches:
        # Return the last match as it's often the YouTube ID
        return matches[-1].strip()

    # If no brackets found, return the stem
    return Path(filename).stem

def has_matching_vtt(audio_file: Path) -> bool:
    """Check if audio file has a matching VTT file."""
    vtt_file = audio_file.with_suffix('.vtt')
    tr_vtt_file = audio_file.with_suffix('.tr.vtt')
    return vtt_file.exists() or tr_vtt_file.exists()

def find_duplicates(directory: Path, extensions: list[str]) -> dict:
    """Find duplicate files based on YouTube IDs."""
    duplicates = defaultdict(list)

    for ext in extensions:
        for file in directory.rglob(f"*.{ext}"):
            youtube_id = extract_youtube_id(file.name)
            if youtube_id:
                key = (youtube_id, file.suffix.lower())
                duplicates[key].append(file)

    return {k: v for k, v in duplicates.items() if len(v) > 1}

def move_duplicates(duplicates: dict, output_dir: Path, console: Console) -> None:
    """Move duplicate files to output directory while preserving structure."""
    for (youtube_id, ext), files in duplicates.items():
        # Sort files by existence of matching VTT
        files.sort(key=has_matching_vtt, reverse=True)

        # Keep the first file (preferably one with matching VTT)
        keep_file = files[0]
        duplicate_files = files[1:]

        console.print(f"\n[cyan]Processing {youtube_id}{ext}[/]")
        console.print(f"[green]Keeping:[/] {keep_file}")

        if has_matching_vtt(keep_file):
            console.print("[green]  (Has matching VTT file)[/]")

        if not duplicate_files:
            continue

        console.print("[yellow]Moving duplicates:[/]")
        for file in track(duplicate_files, description="Moving files"):
            # Preserve directory structure relative to original location
            rel_path = file.parent.relative_to(file.parent.anchor)
            new_path = output_dir / rel_path / file.name

            # Create parent directories
            new_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                # Move file to new location
                shutil.move(str(file), str(new_path))
                console.print(f"  [yellow]Moved:[/] {file} -> {new_path}")

                # Move matching VTT files if they exist
                vtt_file = file.with_suffix('.vtt')
                tr_vtt_file = file.with_suffix('.tr.vtt')

                if vtt_file.exists():
                    vtt_new_path = new_path.with_suffix('.vtt')
                    shutil.move(str(vtt_file), str(vtt_new_path))
                    console.print(f"  [yellow]Moved VTT:[/] {vtt_file.name}")

                if tr_vtt_file.exists():
                    tr_vtt_new_path = new_path.with_suffix('.tr.vtt')
                    shutil.move(str(tr_vtt_file), str(tr_vtt_new_path))
                    console.print(f"  [yellow]Moved TR.VTT:[/] {tr_vtt_file.name}")

            except Exception as e:
                console.print(f"  [red]Error moving {file}: {e}[/]")

def save_duplicate_info(duplicates: dict, input_path: Path, is_folders: bool = False) -> Path:
    """Save duplicate information to a JSON file.

    Args:
        duplicates: Dictionary of duplicates (either files or folders)
        input_path: Base directory path
        is_folders: True if saving folder duplicates, False for file duplicates
    """
    if is_folders:
        # For folders, just use YouTube ID as key
        json_data = {
            youtube_id: [str(f) for f in folders]
            for youtube_id, folders in duplicates.items()
        }
    else:
        # For files, use youtube_id_ext as key
        json_data = {
            f"{youtube_id}_{ext}": [str(f) for f in files]
            for (youtube_id, ext), files in duplicates.items()
        }

    # Save in the input directory
    json_path = input_path / ".duplicates.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    return json_path

def load_duplicate_info(input_path: Path, is_folders: bool = False) -> dict:
    """Load duplicate information from JSON file.

    Args:
        input_path: Base directory path
        is_folders: True if loading folder duplicates, False for file duplicates
    """
    json_path = input_path / ".duplicates.json"
    if not json_path.exists():
        return None

    with open(json_path) as f:
        json_data = json.load(f)

    duplicates = {}
    if is_folders:
        # For folders, just convert paths
        for youtube_id, paths in json_data.items():
            duplicates[youtube_id] = [Path(p) for p in paths]
    else:
        # For files, split key into youtube_id and ext
        for key, paths in json_data.items():
            youtube_id, ext = key.rsplit("_", 1)
            duplicates[(youtube_id, ext)] = [Path(p) for p in paths]

    return duplicates

def get_folder_size(folder: Path) -> int:
    """Calculate total size of a folder in bytes."""
    return sum(f.stat().st_size for f in folder.rglob('*') if f.is_file())

def find_duplicate_folders(directory: Path) -> dict:
    """Find duplicate folders based on YouTube IDs.

    Args:
        directory: Directory to search

    Returns:
        Dictionary mapping YouTube IDs to lists of folder paths
    """
    duplicates = defaultdict(list)
    console = Console()

    # Find all folders recursively
    for folder in directory.rglob('*'):
        if not folder.is_dir() or folder.name == "DUPLICATES":
            continue

        # Extract YouTube ID from folder name
        youtube_id = extract_youtube_id(folder.name)
        if youtube_id:
            # Skip if this is a parent directory of another folder we've already found
            skip = False
            for existing_folder in duplicates[youtube_id]:
                if folder in existing_folder.parents:
                    skip = True
                    break
                if existing_folder in folder.parents:
                    duplicates[youtube_id].remove(existing_folder)
                    break

            if not skip:
                console.print(f"[dim]Found folder: {folder} -> ID: {youtube_id}[/]")
                duplicates[youtube_id].append(folder)
        else:
            console.print(f"[yellow]Warning: Could not extract YouTube ID from folder: {folder.name}[/]")

    # Filter out non-duplicates
    duplicates = {k: v for k, v in duplicates.items() if len(v) > 1}

    # Debug output for found duplicates
    if duplicates:
        console.print("\n[cyan]Found duplicate groups:[/]")
        for youtube_id, folders in duplicates.items():
            console.print(f"[cyan]{youtube_id}:[/]")
            for folder in folders:
                size_mb = get_folder_size(folder) / (1024 * 1024)
                console.print(f"  {folder} ({size_mb:.1f} MB)")

    return duplicates

def move_duplicate_folders(duplicates: dict, output_dir: Path, console: Console) -> None:
    """Move duplicate folders to output directory."""
    for youtube_id, folders in duplicates.items():
        # Sort folders by size (largest first)
        folders.sort(key=get_folder_size, reverse=True)

        # Keep the largest folder
        keep_folder = folders[0]
        duplicate_folders = folders[1:]

        keep_size = get_folder_size(keep_folder) / (1024 * 1024)  # Convert to MB
        console.print(f"\n[cyan]Processing {youtube_id}[/]")
        console.print(f"[green]Keeping:[/] {keep_folder} ({keep_size:.1f} MB)")

        if not duplicate_folders:
            continue

        console.print("[yellow]Moving duplicates:[/]")
        for folder in track(duplicate_folders, description="Moving folders"):
            folder_size = get_folder_size(folder) / (1024 * 1024)  # Convert to MB

            # Create new path preserving original name
            new_path = output_dir / folder.name

            # Handle name conflicts
            if new_path.exists():
                i = 1
                while True:
                    new_path = output_dir / f"{folder.name}_{i}"
                    if not new_path.exists():
                        break
                    i += 1

            try:
                # Move folder to new location
                shutil.move(str(folder), str(new_path))
                console.print(f"  [yellow]Moved:[/] {folder.name} ({folder_size:.1f} MB) -> {new_path}")
            except Exception as e:
                console.print(f"  [red]Error moving {folder}: {e}[/]")

@click.group()
def duplicates():
    """Commands for managing duplicate files."""
    pass

@duplicates.command(name='find')
@click.argument("input_dir", type=click.Path(exists=True))
@click.option("--extensions", "-e", default="m4a,mp3,wav",
              help="Comma-separated list of file extensions to check (default: m4a,mp3,wav)")
def find_duplicates_command(input_dir: str, extensions: str):
    """Find duplicate files based on YouTube IDs in filenames."""
    console = Console()
    input_path = Path(input_dir)
    ext_list = [ext.strip() for ext in extensions.split(",")]

    console.print(f"\nSearching for duplicates in: [blue]{input_path}[/]")
    console.print(f"Checking extensions: [green]{', '.join(ext_list)}[/]\n")

    duplicates = find_duplicates(input_path, ext_list)

    if not duplicates:
        console.print("[green]No duplicates found![/]")
        return

    # Save results
    json_path = save_duplicate_info(duplicates, input_path)
    console.print(f"\nSaved duplicate information to: [blue]{json_path}[/]")

    # Display results
    table = Table(title="Duplicate Files")
    table.add_column("YouTube ID", style="cyan")
    table.add_column("Type", style="yellow")
    table.add_column("Count", style="magenta")
    table.add_column("Files", style="green")

    sorted_keys = sorted(duplicates.keys(), key=lambda x: (x[0], x[1]))

    for youtube_id, ext in sorted_keys:
        files = duplicates[(youtube_id, ext)]
        paths = "\n".join(str(f.relative_to(input_path)) for f in files)
        table.add_row(youtube_id, ext, str(len(files)), paths)

    console.print(table)
    console.print(f"\nFound [red]{len(duplicates)}[/] duplicate file groups")
    console.print("\nRun [cyan]asrtk duplicates move[/] to move duplicates to a separate directory")

@duplicates.command(name='move')
@click.argument("input_dir", type=click.Path(exists=True))
@click.option("--extensions", "-e", default="m4a,mp3,wav",
              help="Comma-separated list of file extensions to check (default: m4a,mp3,wav)")
@click.option("--output-dir", "-o", default="DUPLICATES",
              help="Output directory for duplicates (default: DUPLICATES)")
@click.option("--rescan", is_flag=True, help="Force rescan instead of using saved results")
def move_duplicates_command(input_dir: str, extensions: str, output_dir: str, rescan: bool):
    """Move duplicate files to a separate directory.

    Keeps the first found file with matching VTT and moves others to the output directory.
    Preserves original directory structure in the output directory.
    """
    console = Console()
    input_path = Path(input_dir)
    output_path = Path(input_dir) / output_dir
    ext_list = [ext.strip() for ext in extensions.split(",")]

    # Try to load saved results first
    duplicates = None if rescan else load_duplicate_info(input_path)

    if duplicates is None:
        console.print("[yellow]No saved duplicate information found, scanning...[/]")
        console.print(f"\nSearching for duplicates in: [blue]{input_path}[/]")
        console.print(f"Checking extensions: [green]{', '.join(ext_list)}[/]\n")
        duplicates = find_duplicates(input_path, ext_list)
    else:
        console.print("[green]Using saved duplicate information[/]")

    if not duplicates:
        console.print("[green]No duplicates found![/]")
        return

    # Show what will be moved
    console.print(f"\nOutput directory: [blue]{output_path}[/]")
    total_files = sum(len(files) - 1 for files in duplicates.values())
    console.print(f"\nWill move [red]{total_files}[/] duplicate files to {output_path}")
    console.print("[yellow]Original directory structure will be preserved[/]")
    console.print("[green]Files with matching VTT will be kept in place when possible[/]")

    if not click.confirm("\nDo you want to continue?"):
        console.print("[yellow]Operation cancelled[/]")
        return

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Move duplicates
    move_duplicates(duplicates, output_path, console)

    # Clean up saved results
    saved_results = input_path / ".duplicates.json"
    if saved_results.exists():
        saved_results.unlink()
        console.print("\n[yellow]Cleaned up saved duplicate information[/]")

    console.print("\n[green]Duplicate management complete![/]")

@duplicates.command(name='find-folders')
@click.argument("input_dir", type=click.Path(exists=True))
def find_duplicate_folders_command(input_dir: str):
    """Find duplicate output folders based on YouTube IDs in folder names."""
    console = Console()
    input_path = Path(input_dir)

    console.print(f"\nSearching for duplicate folders in: [blue]{input_path}[/]")

    duplicates = find_duplicate_folders(input_path)

    if not duplicates:
        console.print("[green]No duplicate folders found![/]")
        return

    # Save results with is_folders=True
    json_path = save_duplicate_info(duplicates, input_path, is_folders=True)
    console.print(f"\nSaved duplicate information to: [blue]{json_path}[/]")

    # Display results
    table = Table(title="Duplicate Folders")
    table.add_column("YouTube ID", style="cyan")
    table.add_column("Count", style="magenta")
    table.add_column("Folders (Size)", style="green")

    for youtube_id, folders in sorted(duplicates.items()):
        folder_info = "\n".join(
            f"{f.name} ({get_folder_size(f)/(1024*1024):.1f} MB)"
            for f in folders
        )
        table.add_row(youtube_id, str(len(folders)), folder_info)

    console.print(table)
    console.print(f"\nFound [red]{len(duplicates)}[/] YouTube IDs with duplicate folders")
    console.print("\nRun [cyan]asrtk duplicates move-folders[/] to move duplicate folders")

@duplicates.command(name='move-folders')
@click.argument("input_dir", type=click.Path(exists=True))
@click.option("--output-dir", "-o", default="DUPLICATES",
              help="Output directory for duplicates (default: DUPLICATES)")
@click.option("--rescan", is_flag=True, help="Force rescan instead of using saved results")
def move_duplicate_folders_command(input_dir: str, output_dir: str, rescan: bool):
    """Move duplicate folders to a separate directory."""
    console = Console()
    input_path = Path(input_dir)
    output_path = Path(input_dir) / output_dir

    # Try to load saved results with is_folders=True
    duplicates = None if rescan else load_duplicate_info(input_path, is_folders=True)

    if duplicates is None:
        console.print("[yellow]No saved duplicate information found, scanning...[/]")
        console.print(f"\nSearching for duplicates in: [blue]{input_path}[/]")
        duplicates = find_duplicate_folders(input_path)
    else:
        console.print("[green]Using saved duplicate information[/]")

    if not duplicates:
        console.print("[green]No duplicate folders found![/]")
        return

    # Show what will be moved
    console.print(f"\nOutput directory: [blue]{output_path}[/]")
    total_folders = sum(len(folders) - 1 for folders in duplicates.values())
    console.print(f"\nWill move [red]{total_folders}[/] duplicate folders to {output_path}")
    console.print("[green]Largest folder of each group will be kept in place[/]")

    if not click.confirm("\nDo you want to continue?"):
        console.print("[yellow]Operation cancelled[/]")
        return

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Move duplicates
    move_duplicate_folders(duplicates, output_path, console)

    # Clean up saved results
    saved_results = input_path / ".duplicates.json"
    if saved_results.exists():
        saved_results.unlink()
        console.print("\n[yellow]Cleaned up saved duplicate information[/]")

    console.print("\n[green]Duplicate folder management complete![/]")
