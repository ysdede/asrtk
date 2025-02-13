from pathlib import Path
import pandas as pd
from collections import defaultdict
import argparse
import shutil

def collect_text_files(base_dir: str) -> dict:
    """
    Collect all text files from the dataset directory and organize them by split.
    Expects standard structure with tr_dev_0, tr_test_0, tr_train_0, tr_validated_0.

    Args:
        base_dir: Base directory containing the Common Voice dataset

    Returns:
        Dictionary mapping split names to lists of (text_path, text_content)
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        raise ValueError(f"Directory not found: {base_dir}")

    split_files = defaultdict(list)
    expected_splits = ['tr_dev_0', 'tr_test_0', 'tr_train_0', 'tr_validated_0']

    # Process all .txt files
    for txt_file in base_path.rglob("*.txt"):
        split_name = txt_file.parent.name
        if split_name in expected_splits:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            split_files[split_name].append((str(txt_file), content))

    # Verify all expected splits are found
    missing_splits = set(expected_splits) - set(split_files.keys())
    if missing_splits:
        print(f"Warning: Missing splits: {missing_splits}")

    return dict(split_files)

def find_unique_validated(split_files: dict) -> list:
    """
    Find samples in tr_validated_0 that don't appear in other splits (based on filename).

    Args:
        split_files: Dictionary mapping split names to lists of (text_path, text_content)

    Returns:
        List of (path, content) tuples for unique validated samples
    """
    # Get filenames from non-validated splits
    other_files = set()
    for split_name, files in split_files.items():
        if split_name != 'tr_validated_0':
            other_files.update(Path(path).stem for path, _ in files)

    # Find unique validated samples
    validated_unique = []
    if 'tr_validated_0' in split_files:
        for path, content in split_files['tr_validated_0']:
            if Path(path).stem not in other_files:
                validated_unique.append((path, content))

    return validated_unique

def extract_unique_validated(unique_samples: list, output_dir: str):
    """
    Extract unique validated samples to a new directory.

    Args:
        unique_samples: List of (path, content) tuples
        output_dir: Directory to save the extracted files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Starting to copy {len(unique_samples)} files...")
    copied = 0
    errors = []

    for orig_path, _ in unique_samples:
        try:
            txt_path = Path(orig_path)
            audio_path = txt_path.with_suffix('.mp3')

            # Skip if source files don't exist
            if not txt_path.exists() or not audio_path.exists():
                raise FileNotFoundError(f"Missing source file for {txt_path.stem}")

            # Copy files
            shutil.copy2(txt_path, output_path / txt_path.name)
            shutil.copy2(audio_path, output_path / audio_path.name)
            copied += 1

            if copied % 1000 == 0:
                print(f"Copied {copied} pairs of files...")

        except Exception as e:
            errors.append(f"Error processing {orig_path}: {str(e)}")

    print(f"\nCopying complete! Successfully copied {copied} pairs of files")
    if errors:
        print(f"\nEncountered {len(errors)} errors:")
        for error in errors[:10]:
            print(error)
        if len(errors) > 10:
            print(f"...and {len(errors) - 10} more errors")

def analyze_splits(split_files: dict) -> pd.DataFrame:
    """
    Analyze splits for duplicates based on filenames.

    Args:
        split_files: Dictionary mapping split names to lists of (text_path, text_content)

    Returns:
        DataFrame containing duplicate analysis
    """
    # Create sets of filenames for each split
    split_files_dict = {
        split: {Path(path).stem for path, _ in files}
        for split, files in split_files.items()
    }

    analysis_data = []
    splits = sorted(split_files.keys())  # Sort for consistent output

    for i, split1 in enumerate(splits):
        for split2 in splits[i+1:]:
            intersection = split_files_dict[split1] & split_files_dict[split2]
            if intersection:
                examples = []
                for filename in sorted(intersection)[:3]:
                    path1 = next(p for p, _ in split_files[split1] if Path(p).stem == filename)
                    path2 = next(p for p, _ in split_files[split2] if Path(p).stem == filename)
                    content = next(c for _, c in split_files[split1] if Path(path1).stem == filename)
                    examples.append(f"\n{filename} (text: {content})\n→ {path1}\n→ {path2}")

                analysis_data.append({
                    'Split 1': split1,
                    'Split 2': split2,
                    'Duplicate Count': len(intersection),
                    'Split 1 Total': len(split_files[split1]),
                    'Split 2 Total': len(split_files[split2]),
                    'Example Duplicates': '\n'.join(examples)
                })

    return pd.DataFrame(analysis_data)

def main():
    parser = argparse.ArgumentParser(description='Analyze CommonVoice dataset splits and extract unique validated samples')
    parser.add_argument('input_dir', type=str, help='Input directory containing CommonVoice dataset')
    parser.add_argument('--output', '-o', type=str, default='commonvoice_duplicates_analysis.csv',
                      help='Output CSV file path (default: commonvoice_duplicates_analysis.csv)')
    parser.add_argument('--extract-dir', '-e', type=str,
                      help='Directory to extract unique validated samples (optional)')
    parser.add_argument('--copy-unique', '-c', action='store_true',
                      help='Copy unique files from validated splits to extract directory')

    args = parser.parse_args()

    print(f"Analyzing CommonVoice dataset in {args.input_dir}")
    split_files = collect_text_files(args.input_dir)

    # Print basic statistics
    print("\nFiles found per split:")
    for split, files in sorted(split_files.items()):
        print(f"{split}: {len(files)} files")

    # Analyze duplicates
    print("\nAnalyzing duplicates between splits...")
    analysis_df = analyze_splits(split_files)
    analysis_df.to_csv(args.output, index=False)
    print(f"Analysis saved to {args.output}")

    # Print duplicate summary
    if len(analysis_df) == 0:
        print("\nNo duplicates found between splits!")
    else:
        print("\nDuplicate Summary:")
        for _, row in analysis_df.iterrows():
            print(f"\n{row['Split 1']} vs {row['Split 2']}:")
            print(f"Found {row['Duplicate Count']} duplicates")
            print("Example duplicates:", row['Example Duplicates'])

    # Find unique validated samples using the duplicate analysis
    if 'tr_validated_0' in split_files:
        # Get all filenames that appear in duplicates
        duplicate_files = set()
        for _, row in analysis_df.iterrows():
            if row['Split 1'] == 'tr_validated_0' or row['Split 2'] == 'tr_validated_0':
                split1_files = {Path(p).stem for p, _ in split_files[row['Split 1']]}
                split2_files = {Path(p).stem for p, _ in split_files[row['Split 2']]}
                duplicate_files.update(split1_files & split2_files)

        # Find unique files (those not in duplicates)
        unique_validated = [
            (path, content) for path, content in split_files['tr_validated_0']
            if Path(path).stem not in duplicate_files
        ]

        print(f"\nFound {len(unique_validated)} unique samples in tr_validated_0")

        # Copy unique files if requested
        if args.copy_unique and args.extract_dir:
            print(f"\nExtracting unique validated samples to {args.extract_dir}...")
            extract_unique_validated(unique_validated, args.extract_dir)

if __name__ == "__main__":
    main()
