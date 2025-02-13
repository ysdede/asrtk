import json
import logging
from pathlib import Path
import webvtt
from datasets import Dataset, Audio, Value, Features
import os

# Replace magic command with os.chdir for .py file
os.chdir("N:/dataset_v3/YENI_SPLIT")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_metadata_vtt(vtt_path):
    """Extract text from WebVTT file, combining all captions into one transcription."""
    try:
        captions = webvtt.read(str(vtt_path))
        # Combine all caption texts into one string, removing any newlines
        transcription = ' '.join(caption.text.replace('\n', ' ') for caption in captions)
        return {
            'transcription': transcription.strip()
        }
    except Exception as e:
        logging.error(f"Error reading VTT file {vtt_path}: {str(e)}")
        return None

def get_metadata_json(json_path):
    """Extract text from JSON file."""
    try:
        with open(json_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON file: {json_path}")
        return None

    text = data.get('text', '')
    text_norm = data.get('text_normalized', '')
    transcription = text or text_norm

    if not transcription:
        logging.warning(f"No transcription found in JSON file: {json_path}")
        return None

    return {'transcription': transcription.strip()}

def scan_for_files(directory, audio_ext='.mp3', metadata_ext='.json'):
    """
    Scan directory for metadata and audio file pairs.

    Args:
        directory (str or Path): Directory to scan
        audio_ext (str): Audio file extension (default: '.mp3')
        metadata_ext (str): Metadata file extension (default: '.json')
    """
    directory = Path(directory)
    logging.info(f"Scanning for {metadata_ext} files in {directory}...")
    data = []

    # Get metadata handler based on extension
    metadata_handlers = {
        '.vtt': get_metadata_vtt,
        '.json': get_metadata_json
    }

    if metadata_ext not in metadata_handlers:
        raise ValueError(f"Unsupported metadata extension: {metadata_ext}. Supported types: {list(metadata_handlers.keys())}")

    metadata_handler = metadata_handlers[metadata_ext]

    try:
        # Use os.walk instead of rglob to better handle special characters
        for root, _, files in os.walk(str(directory)):
            for file in files:
                if file.endswith(metadata_ext):
                    metadata_path = Path(os.path.join(root, file))
                    audio_path = metadata_path.with_suffix(audio_ext)

                    if not audio_path.exists():
                        logging.warning(f"No matching audio file found for: {metadata_path}")
                        continue

                    # Get metadata
                    metadata = metadata_handler(metadata_path)
                    if not metadata:
                        continue

                    if len(metadata['transcription']) < 5:
                        continue

                    metadata['audio'] = {'path': str(audio_path)}
                    data.append(metadata)

        logging.info(f"Found {len(data)} valid file pairs")
        return data

    except Exception as e:
        logging.error(f"Error scanning directory: {str(e)}")
        return []

def create_dataset(data_dir, audio_ext='.mp3', metadata_ext='.json'):
    """
    Create dataset from directory containing metadata and audio files.

    Args:
        data_dir (str or Path): Directory containing the data
        audio_ext (str): Audio file extension to look for (default: '.mp3')
        metadata_ext (str): Metadata file extension to look for (default: '.json')
    """
    data = scan_for_files(data_dir, audio_ext, metadata_ext)

    features = Features({
        "audio": Audio(sampling_rate=16_000),
        "transcription": Value("string")
    })

    dataset = Dataset.from_list(data, features=features)
    # Split the dataset into train and test
    train_test_split = dataset.train_test_split(test_size=0.05, seed=42)
    return train_test_split

# Example usage
train_test_dataset = create_dataset(
    "N:/dataset_v3/YENI_SPLIT",
    audio_ext='.mp3',  # or '.opus'
    metadata_ext='.vtt'  # or '.vtt'
)

# Save the dataset with both splits to disk
train_test_dataset.save_to_disk(str(Path('ysdede/yeni-split-0').resolve()))
