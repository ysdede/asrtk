import os
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def cleanup_unpaired_files(directory, extensions=('.mp3', '.vtt')):
    """
    Scan directory for files with given extensions and remove files without pairs.

    Args:
        directory (str or Path): Directory to scan
        extensions (tuple): Tuple of extension pairs to check (default: ('.mp3', '.vtt'))
    """
    directory = Path(directory)
    files_removed = 0

    try:
        # Walk through all subdirectories
        for root, _, files in os.walk(str(directory)):
            root_path = Path(root)

            # Group files by their stem (filename without extension)
            file_groups = {}
            for file in files:
                if file.endswith(extensions):
                    stem = Path(file).stem
                    if stem not in file_groups:
                        file_groups[stem] = []
                    file_groups[stem].append(file)

            # Check each group for unpaired files
            for stem, group in file_groups.items():
                if len(group) != len(extensions):
                    # This group has unpaired files
                    for file in group:
                        file_path = root_path / file
                        logging.info(f"Removing unpaired file: {file_path}")
                        file_path.unlink()  # Delete the file
                        files_removed += 1

        logging.info(f"Cleanup complete. Removed {files_removed} unpaired files.")

    except Exception as e:
        logging.error(f"Error during cleanup: {str(e)}")

if __name__ == "__main__":
    # Set your directory path here
    directory = "N:/dataset_v3/YENI_SPLIT_LQ_NOISY"

    # Confirm before proceeding
    print(f"This will remove all .mp3 and .vtt files that don't have matching pairs in:")
    print(f"{directory}")
    print("\nAre you sure you want to continue? This cannot be undone!")
    response = input("Type 'yes' to continue: ")

    if response.lower() == 'yes':
        cleanup_unpaired_files(directory)
    else:
        print("Operation cancelled.")
