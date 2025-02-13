from pathlib import Path
import json
import argparse

def process_metadata_file(input_path: Path, output_path: Path):
    """
    Process a metadata JSON file to truncate paths to filenames.

    Args:
        input_path: Path to input metadata JSON file
        output_path: Path to save processed metadata
    """
    print(f"Processing {input_path.name}...")

    # Read input JSON
    with open(input_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    # Process each entry
    for entry in metadata:
        # Convert path to just filename
        entry['path'] = Path(entry['path']).name

    # Save processed metadata
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Saved processed metadata to {output_path}")
    print(f"Processed {len(metadata)} entries")

def main():
    parser = argparse.ArgumentParser(description='Process CommonVoice metadata files to truncate paths')
    parser.add_argument('input_dir', type=str, help='Directory containing metadata JSON files')
    parser.add_argument('--output-dir', '-o', type=str, help='Output directory (default: input_dir + "_processed")')

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise ValueError(f"Input directory not found: {input_dir}")

    # Set up output directory
    output_dir = Path(args.output_dir) if args.output_dir else input_dir.parent / f"{input_dir.name}_processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process all JSON files
    json_files = list(input_dir.glob('*_metadata.json'))
    if not json_files:
        print(f"No metadata files found in {input_dir}")
        return

    print(f"Found {len(json_files)} metadata files")
    for json_file in json_files:
        output_path = output_dir / json_file.name
        process_metadata_file(json_file, output_path)

    print("\nAll files processed!")

if __name__ == "__main__":
    main()
