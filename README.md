# ASRTK: Automatic Speech Recognition Toolkit

ASRTK is a Python toolkit focused on collecting and processing high-quality audio datasets for training Automatic Speech Recognition (ASR) systems. It provides a streamlined workflow and utilities for downloading, cleaning, and preparing audio data with accurate transcriptions.

## Key Features

### Data Collection & Preparation
- **YouTube Integration**: Automated download of videos with subtitles from playlists
- **Audio Processing**: Advanced audio splitting, conversion, and resampling capabilities
- **Subtitle Processing**: Comprehensive VTT file handling with support for cleaning and normalization

### Text Processing & Analysis
- **Text Normalization**: Intelligent handling of abbreviations, numbers, and special characters
- **Pattern Analysis**: Tools for analyzing word frequencies and text patterns

### Audio Processing
- **Forced Alignment**: Precise audio-text alignment using Silero VAD models
- **Format Optimization**: Efficient audio format conversion with multi-threading support
- **Quality Control**: Advanced handling of silence detection and failed alignments

### Workflow Automation
- **Batch Processing**: Efficient handling of large datasets
- **Pipeline Integration**: Seamless workflow from data collection to model training
- **Progress Tracking**: Detailed logging and progress monitoring

## Requirements

- Python 3.9 or higher
- PyTorch 1.7 or higher
- NumPy
- SciPy
- librosa
- pydub
- transformers
- BeautifulSoup4
- Additional dependencies in `requirements.txt`

## Installation

```bash
# Clone the repository
git clone https://github.com/ysdede/asrtk.git

# Navigate to project directory
cd asrtk

# Install dependencies
pip install -r requirements.txt
```

## Quick Start Guide

### 1. Download Training Data
```bash
# Download from a single playlist
asrtk download-playlist ./dataset "https://www.youtube.com/playlist?list=PLAYLIST_ID"

# Or download from multiple playlists
asrtk download-playlist ./dataset --playlist-file playlists.txt
```

### 2. Process Audio Files
```bash
# Split audio into chunks based on subtitles
asrtk split ./dataset ./chunks --tolerance 500 -fm --restore-punctuation

# Convert to optimized format
asrtk convert-opus ./chunks --remove-original --workers 8
```

### 3. Clean and Normalize Text
```bash
# Apply text normalization rules
asrtk apply-patch rules.json ./chunks

# Analyze text patterns
asrtk create-wordset ./chunks --min-frequency 5
```

## Advanced Usage

### Audio Processing Options
```bash
asrtk split INPUT_DIR OUTPUT_DIR [OPTIONS]
  --tolerance MILLISECONDS    Alignment tolerance
  --force-merge, -fm         Merge consecutive subtitles
  --restore-punctuation      Use BERT for punctuation restoration
  --keep-effects            Preserve effect annotations
```

### Text Processing Features
```bash
asrtk apply-patch RULES INPUT_DIR [OPTIONS]
  --dry-run                 Preview changes
  --show-replacements       Display detailed replacement plan
  --verbose                Show per-file changes
```

### Batch Processing
```bash
asrtk convert-opus INPUT_DIR [OPTIONS]
  --input-type FORMAT       Source audio format
  --remove-original        Delete source files after conversion
  --workers NUMBER         Parallel processing threads
```

## Project Structure
```
asrtk/
├── dataset/               # Downloaded content
│   ├── json_info/        # Metadata cache
│   └── content/          # Audio and subtitle files
├── chunks/               # Processed audio segments
└── output/              # Final processed dataset
```

## Contributing

We welcome contributions! Please feel free to submit pull requests or open issues for:
- Bug fixes
- Feature additions
- Documentation improvements
- Performance optimizations

## License

This project is licensed under the MIT License - see the LICENSE file for details.

