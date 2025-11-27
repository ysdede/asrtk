from datasets import load_dataset, Audio
from pathlib import Path
import os
import tqdm
import json
import requests
import shutil
from dotenv import load_dotenv

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

def save_metadata(metadata_list, output_dir: Path, split: str):
    """Save dataset metadata to JSON file.

    Args:
        metadata_list: List of metadata dictionaries
        output_dir: Output directory path
        split: Dataset split name
    """
    metadata_file = output_dir / f"{split}_metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, ensure_ascii=False, indent=2)

    print(f"Saved metadata for {len(metadata_list)} items to {metadata_file}")

def download_dataset(dataset_name: str, subset: str, lang: str, output_dir: str, splits: list[str]):
    """Download/extract and organize dataset with audio files and metadata, avoiding re-encoding.

    Audio is extracted by loading the dataset with Audio(decode=False) to get raw bytes,
    and then writing these bytes to a file using the original extension.
    If audio is a remote URL, it's downloaded directly.

    Args:
        dataset_name: Name of the dataset on HuggingFace
        subset: Dataset subset (can be None)
        lang: Language code (can be None)
        output_dir: Output directory path
        splits: List of splits to download (e.g., ['train', 'test', 'validation'])
    """
    print(f"Loading {dataset_name} dataset...")
    if subset:
        print(f"Loading {subset} subset...")
    if lang:
        print(f"Loading {lang} language...")
    print(f"Loading {splits} splits...")

    base_dir = setup_directory(output_dir)

    for split in splits:
        print(f"\nProcessing {split} split...")

        split_dir = base_dir / split
        split_dir.mkdir(exist_ok=True)

        dataset_args = [dataset_name]
        if lang:
            dataset_args.append(lang)
        if subset:
            dataset_args.append(subset)

        print("Loading dataset and casting audio to raw bytes (decode=False)...")
        try:
            dataset = load_dataset(
                *dataset_args,
                split=split,
                streaming=False, # Ensures data is downloaded/cached
                trust_remote_code=True
            ).cast_column("audio", Audio(decode=False)) # Key change for raw audio
        except Exception as e:
            print(f"Error loading dataset {dataset_name} ({split}): {e}")
            print("Please ensure the dataset has an 'audio' column.")
            continue

        print(f"Dataset columns: {dataset.column_names}")
        print(f"Dataset size: {len(dataset)}")

        metadata_list = []
        print("Extracting audio files (original format) and collecting metadata...")

        for idx, item in enumerate(tqdm.tqdm(dataset)):
            try:
                audio_data = item.get('audio')
                if not audio_data:
                    print(f"Warning: No audio data found for item {idx}. Skipping.")
                    continue

                # Determine audio file extension from original path
                audio_ext = None
                original_path_for_ext = audio_data.get('path')

                if isinstance(original_path_for_ext, str) and original_path_for_ext:
                    audio_ext = Path(original_path_for_ext).suffix

                # If extension is missing or path is None, try specific dataset knowledge or default
                if not audio_ext:
                    if "ysdede/khanacademy-turkish" in dataset_name and audio_data.get('bytes'):
                        audio_ext = ".opus" # Specific knowledge for this dataset
                    elif audio_data.get('bytes'): # If we have bytes but no extension info
                        audio_ext = ".bin" # Generic binary extension
                        print(f"Warning: Item {idx} has audio bytes but unknown original extension. Saving as .bin. Original path: {original_path_for_ext}")
                    else: # No bytes and no path extension
                         print(f"Warning: Could not determine audio extension for item {idx}. Skipping. Original path: {original_path_for_ext}")
                         continue

                if not audio_ext.startswith('.'): # Ensure extension starts with a dot
                    audio_ext = '.' + audio_ext

                base_filename = f"{split}_{idx:06d}"
                audio_filename = f"{base_filename}{audio_ext}"
                audio_path = split_dir / audio_filename

                audio_saved = False
                if not audio_path.exists():
                    # Priority 1: Save from raw bytes (if available)
                    raw_bytes = audio_data.get('bytes')
                    if raw_bytes is not None:
                        try:
                            with open(audio_path, 'wb') as f:
                                f.write(raw_bytes)
                            audio_saved = True
                        except Exception as e:
                            print(f"Error saving audio bytes for item {idx} to {audio_path}: {e}")

                    # Priority 2: If no bytes, but path is a downloadable URL
                    elif isinstance(original_path_for_ext, str) and original_path_for_ext.startswith(('http://', 'https://')):
                        print(f"Note: Item {idx} has no raw bytes, attempting download from URL: {original_path_for_ext}")
                        try:
                            response = requests.get(original_path_for_ext, stream=True)
                            response.raise_for_status()
                            with open(audio_path, 'wb') as f:
                                shutil.copyfileobj(response.raw, f)
                            audio_saved = True
                        except Exception as e:
                            print(f"Error downloading audio from URL for item {idx} ({original_path_for_ext}): {e}")

                    # Priority 3: If path is an absolute local file path (and bytes weren't loaded for some reason)
                    elif isinstance(original_path_for_ext, str) and Path(original_path_for_ext).is_file():
                        print(f"Note: Item {idx} has no raw bytes, attempting copy from local file path: {original_path_for_ext}")
                        try:
                            shutil.copy2(original_path_for_ext, audio_path)
                            audio_saved = True
                        except Exception as e:
                            print(f"Error copying local audio file for item {idx} ({original_path_for_ext}): {e}")


                if not audio_saved and not audio_path.exists():
                    print(f"Warning: Could not save audio for item {idx}. No bytes, URL, or valid local file. Skipping.")
                    continue

                metadata_item = {
                    'audio_file': audio_filename,
                    'audio_path': str(audio_path.relative_to(base_dir)),
                    'index': idx,
                    'original_source_path': original_path_for_ext # Log original path for reference
                }

                text_fields = ['sentence', 'transcription', 'text', 'transcript']
                for field in text_fields:
                    if field in item and item[field] is not None:
                        metadata_item[field] = item[field]

                useful_fields = ['speaker_id', 'gender', 'age', 'accent', 'duration', 'sampling_rate']
                for field in useful_fields:
                    if field in item and item[field] is not None: # 'duration' and 'sampling_rate' might not be in item if not decoded
                        metadata_item[field] = item[field]
                    elif field in audio_data and audio_data[field] is not None: # Check audio_data dict too
                         metadata_item[field] = audio_data[field]

                metadata_list.append(metadata_item)

            except Exception as e:
                print(f"Error processing item {idx}: {str(e)}")
                continue

        save_metadata(metadata_list, base_dir, split)

def main():
    from huggingface_hub import login

    load_dotenv()
    hf_token = os.getenv('HF_TOKEN')

    if hf_token:
        login(token=hf_token)
        print("Logged in to HuggingFace using token from .env file")
    else:
        print("Warning: No HF_TOKEN found in .env file. Some datasets may not be accessible.")

    # DATASET_NAME = "ysdede/khanacademy-turkish"
    # DATASET_NAME = "ysdede/commonvoice_17_tr_fixed"
    # DATASET_NAME = "ysdede/yeni-split-0"
    # DATASET_NAME = "ysdede/yeni-split-lq-noisy"
    # DATASET_NAME = "erenfazlioglu/turkishvoicedataset"
    DATASET_NAME = "ysdede/tr-med-audio"
    SUBSET = None
    LANG = None
    OUTPUT_DIR = "N:/dataset_clean/" + DATASET_NAME.replace("/", "_") # Ensure valid directory name

    SPLITS = ['train']

    try:
        download_dataset(DATASET_NAME, SUBSET, LANG, OUTPUT_DIR, SPLITS)

        print("\nDownload completed successfully!")

        print("\nDataset structure:")
        base_dir = Path(OUTPUT_DIR)

        for split_name in SPLITS:
            current_split_dir = base_dir / split_name
            if current_split_dir.exists():
                # Glob for any common audio extension, not just one type
                audio_files_in_split = [p for p in current_split_dir.iterdir() if p.suffix.lower() in ('.wav', '.mp3', '.opus', '.flac', '.ogg', '.bin')]
                metadata_file_path = base_dir / f"{split_name}_metadata.json"

                print(f"\n{split_name.upper()} split:")
                print(f"  Audio files: {len(audio_files_in_split)}")
                print(f"  Metadata file: {metadata_file_path}")

                if metadata_file_path.exists():
                    with open(metadata_file_path, 'r', encoding='utf-8') as f:
                        loaded_metadata = json.load(f)
                    print(f"  Metadata entries: {len(loaded_metadata)}")

                    if loaded_metadata:
                        print(f"  Sample metadata keys: {list(loaded_metadata[0].keys())}")

        total_audio_files_downloaded = sum(len(list((base_dir / s).glob("*.*"))) for s in SPLITS if (base_dir / s).exists()) # A rough count
        print(f"\nTotal files in output splits: {total_audio_files_downloaded}")

    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
