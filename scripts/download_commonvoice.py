from datasets import load_dataset
from pathlib import Path
import os
import tqdm
import json
import requests

def setup_directory(base_dir: str) -> Path:
    """Create base directory for dataset.

    Args:
        base_dir: Base directory path

    Returns:
        Path: Base directory path
    """
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    return base_path

def save_metadata(dataset, output_dir: Path, split: str):
    """Save dataset metadata to JSON file.

    Args:
        dataset: Dataset split
        output_dir: Output directory path
        split: Dataset split name
    """
    metadata = []
    for item in dataset:
        metadata.append({
            'path': item['path'],
            'sentence': item['sentence'],
            'up_votes': item['up_votes'],
            'down_votes': item['down_votes'],
            'age': item['age'],
            'gender': item['gender'],
            'accent': item['accent']
        })

    metadata_file = output_dir / f"{split}_metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

def download_dataset(output_dir: str, splits: list[str]):
    """Download and organize Common Voice dataset.

    Args:
        output_dir: Output directory path
        splits: List of splits to download (e.g., ['train', 'test', 'validation'])
    """
    print("Loading Common Voice 17.0 Turkish dataset...")
    base_dir = setup_directory(output_dir)

    # Load dataset for each split
    for split in splits:
        print(f"\nProcessing {split} split...")

        # Load the dataset split with trust_remote_code=True
        dataset = load_dataset(
            "mozilla-foundation/common_voice_17_0",
            "tr",
            split=split,
            streaming=False,
            trust_remote_code=True
        ).remove_columns("audio")  # Remove the audio column to avoid decoding

        # Save metadata
        save_metadata(dataset, base_dir, split)

        # Process each item
        print("Saving audio files and transcripts...")
        for item in tqdm.tqdm(dataset):
            # Get original path and create corresponding directory structure
            orig_path = Path(item['path'])
            rel_dir = orig_path.parent  # e.g., 'tr_validated_2'

            # Create directory if it doesn't exist
            save_dir = base_dir / rel_dir
            save_dir.mkdir(parents=True, exist_ok=True)

            # Generate filenames using original naming
            base_filename = orig_path.stem  # e.g., 'common_voice_tr_40181287'
            audio_path = save_dir / f"{base_filename}.mp3"
            transcript_path = save_dir / f"{base_filename}.txt"

            # Download the audio file directly from the URL
            if not audio_path.exists():  # Skip if already exists
                audio_url = item['audio']['path']  # Get the URL of the audio file
                response = requests.get(audio_url)
                with open(audio_path, 'wb') as f:
                    f.write(response.content)

            # Save transcript
            if not transcript_path.exists():  # Skip if already exists
                with open(transcript_path, 'w', encoding='utf-8') as f:
                    f.write(item['sentence'])

def main():
    # Configuration
    OUTPUT_DIR = "common_voice_tr"
    SPLITS = ['train', 'test', 'validation', 'validated']

    try:
        download_dataset(OUTPUT_DIR, SPLITS)
        print("\nDownload completed successfully!")

        # Print dataset structure
        print("\nDataset structure:")
        base_dir = Path(OUTPUT_DIR)

        # Count files by directory
        dir_counts = {}
        for audio_file in base_dir.rglob("*.mp3"):
            dir_name = audio_file.parent.name
            dir_counts[dir_name] = dir_counts.get(dir_name, 0) + 1

        # Print statistics
        total_files = sum(dir_counts.values())
        print(f"\nTotal audio files: {total_files}")
        print("\nFiles per directory:")
        for dir_name, count in sorted(dir_counts.items()):
            print(f"  {dir_name}: {count} files")

    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
