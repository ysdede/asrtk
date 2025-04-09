from pathlib import Path
import pandas as pd
from collections import defaultdict
import argparse

def collect_labels_from_splits(base_dir: str) -> dict:
    """
    Collect all MP3 file labels from each split directory.

    Args:
        base_dir: Base directory containing the dataset splits

    Returns:
        Dictionary mapping split names to lists of labels found
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        raise ValueError(f"Directory not found: {base_dir}")

    split_labels = defaultdict(list)
    expected_splits = ['train', 'validated', 'validation', 'test']

    # Process all splits
    for split in expected_splits:
        split_path = base_path / split
        if split_path.exists():
            print(f"\nProcessing {split} directory...")
            # Get all MP3 files in this split
            mp3_files = list(split_path.glob("*.mp3"))
            for file_path in mp3_files:
                label = file_path.stem
                split_labels[split].append(label)
            
            print(f"Found {len(mp3_files)} MP3 files in {split}")
            if len(split_labels[split]) > 0:
                print(f"First few labels: {split_labels[split][:3]}")

    # Verify all expected splits are found
    missing_splits = set(expected_splits) - set(split_labels.keys())
    if missing_splits:
        print(f"Warning: Missing splits: {missing_splits}")

    return dict(split_labels)

def analyze_split_leaks(split_labels: dict) -> pd.DataFrame:
    """
    Analyze splits for label leaks (same label appearing in multiple splits).

    Args:
        split_labels: Dictionary mapping split names to lists of labels

    Returns:
        DataFrame containing leak analysis
    """
    analysis_data = []
    splits = sorted(split_labels.keys())

    # For verification
    test_label = "common_voice_tr_28941816"
    
    for i, split1 in enumerate(splits):
        labels1 = set(split_labels[split1])
        
        for split2 in splits[i+1:]:
            labels2 = set(split_labels[split2])
            
            # Find common labels (leaks)
            leaks = labels1 & labels2
            
            # Debug check for our test case
            if test_label in leaks:
                print(f"\nVERIFICATION: Found test label '{test_label}' as leak between {split1} and {split2}")
            
            if leaks:
                example_leaks = sorted(list(leaks))[:5]
                
                analysis_data.append({
                    'Split 1': split1,
                    'Split 2': split2,
                    'Leak Count': len(leaks),
                    'Split 1 Total': len(labels1),
                    'Split 2 Total': len(labels2),
                    'Leak Percentage': round(len(leaks) * 100 / min(len(labels1), len(labels2)), 2),
                    'Example Leaks': '\n'.join(example_leaks),
                    'All Leaks': sorted(list(leaks))
                })
            
            print(f"\nComparing {split1} vs {split2}:")
            print(f"- {split1}: {len(labels1)} files")
            print(f"- {split2}: {len(labels2)} files")
            print(f"- Found {len(leaks)} duplicates")
            if leaks:
                print(f"- First few duplicates: {sorted(list(leaks))[:3]}")

    return pd.DataFrame(analysis_data)

def main():
    parser = argparse.ArgumentParser(description='Analyze dataset splits for MP3 file duplicates')
    parser.add_argument('input_dir', type=str, help='Input directory containing dataset splits')
    parser.add_argument('--output', '-o', type=str, default='dataset_leaks_analysis.csv',
                      help='Output CSV file path (default: dataset_leaks_analysis.csv)')

    args = parser.parse_args()

    print(f"Analyzing dataset splits in {args.input_dir}")
    split_labels = collect_labels_from_splits(args.input_dir)

    # Print basic statistics
    print("\nMP3 files found per split:")
    for split, labels in sorted(split_labels.items()):
        print(f"{split}: {len(labels)} files")

    # Analyze leaks
    print("\nAnalyzing duplicates between splits...")
    analysis_df = analyze_split_leaks(split_labels)
    
    if not analysis_df.empty:
        # Save detailed leaks to a separate file
        detailed_output = args.output.replace('.csv', '_detailed.csv')
        analysis_df.to_csv(detailed_output, index=False)
        
        # Save summary without the full leak lists
        summary_df = analysis_df.drop(columns=['All Leaks'])
        summary_df.to_csv(args.output, index=False)
        
        print(f"\nAnalysis saved to:")
        print(f"- Summary: {args.output}")
        print(f"- Detailed: {detailed_output}")
        
        # Print summary of findings
        total_leaks = sum(row['Leak Count'] for _, row in analysis_df.iterrows())
        print(f"\nTotal number of duplicate files found: {total_leaks}")
    else:
        print("\nNo duplicates found between splits!")

if __name__ == "__main__":
    main()
