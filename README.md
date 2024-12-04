# ASRTK: Automatic Speech Recognition Toolkit

ASRTK is a comprehensive Python toolkit for building and deploying end-to-end Automatic Speech Recognition (ASR) systems.  
This project is a rewrite and open-source release of the previously closed-source toolkit known as "YTK".  
It provides a streamlined workflow and a collection of utilities to simplify the process of creating, training, and evaluating ASR models.

## Key Features

- **Data Preparation**: Easily prepare and preprocess speech datasets, including data cleaning, normalization, and feature extraction.
- *Model Training: Train state-of-the-art ASR models using popular architectures and techniques. (Coming soon)*
- **Evaluation and Testing**: Evaluate the performance of trained models using various metrics and perform inference on new audio data.
- ~~Deployment: Deploy trained ASR models in real-world applications with support for different platforms and environments.~~

## Requirements

- Python 3.9 or higher
- PyTorch 1.7 or higher
- NumPy
- SciPy
- librosa

<!-- start docs-include-installation -->

## Getting Started

To get started with ASRTK, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/asrtk.git
   ```

2. Navigate to the project directory:
   ```bash
   cd asrtk
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

<!-- end docs-include-installation -->

## Usage

<!-- start docs-include-usage -->

Running `asrtk --help` or `python -m asrtk --help` shows a list of all of the available options and commands:

<!-- [[[cog
import cog
from asrtk import cli
from click.testing import CliRunner
runner = CliRunner()
result = runner.invoke(cli.cli, ["--help"], terminal_width=88)
help = result.output.replace("Usage: cli", "Usage: asrtk")
cog.outl(f"\n```sh\nasrtk --help\n{help.rstrip()}\n```\n")
]]] -->
<!-- [[[end]]] -->

<!-- end docs-include-usage -->

## Dataset Creation Workflow

### 1. Download YouTube Content with Subtitles

The first step in creating an ASR dataset is collecting source material. ASRTK provides a command to download YouTube playlists while ensuring the videos have subtitles:

```bash
asrtk download-playlist WORK_DIR [PLAYLIST_URL] [PLAYLIST_FILE]
```

This command:
- Downloads videos from YouTube playlists that have subtitles
- Organizes downloads into playlist-specific directories
- Caches playlist metadata to avoid redundant downloads
- Supports both single playlist URLs or a file containing multiple playlist URLs

Options:
- `WORK_DIR`: Directory where downloads and metadata will be stored
- `PLAYLIST_URL`: (Optional) Direct YouTube playlist URL
- `PLAYLIST_FILE`: (Optional) Path to a file containing playlist URLs (one per line)

Example usage:
```bash
# Download a single playlist
asrtk download-playlist ./my_dataset "https://www.youtube.com/playlist?list=PLAYLIST_ID"

# Download multiple playlists from a file
asrtk download-playlist ./my_dataset --playlist-file playlists.txt
```

The command will create a structured directory containing:
```
my_dataset/
├── json_info/              # Cached playlist metadata
│   └── PLAYLIST_ID.json    # Playlist information cache
│
└── Playlist_Name/          # One directory per playlist
    ├── video1.m4a         # Audio files
    ├── video1.tr.vtt      # Subtitle files
    ├── video2.m4a
    └── video2.tr.vtt
```

This downloaded content serves as the raw material for the next step in the pipeline, where audio tracks and subtitles are processed to create aligned training pairs for ASR systems.

## Contributing

We welcome contributions from the community! Whether it's adding new features, improving documentation, or reporting bugs, please feel free to make a pull request or open an issue.

## License

ASRTK is released under the MIT license. Contributions must adhere to this license.

yt-dlp --flat-playlist --lazy-playlist --print-to-file url r:\youtube\khanAcademi.txt https://www.youtube.com/@**/playlists

asrtk download-playlist r:\youtube\khanAkademiTr  --file r:\youtube\khanAcademi.txt

asrtk split R:\youtube\khanAkademiTr\Depresyon__Bipolar_Bozukluk_ve_Anksiyete___Psikoloji r:\youtube\test_output -fm --tolerance 500

yt-dlp --flat-playlist -J https://www.youtube.com/@esrinart > r:\youtube\esrinart.json

asrtk download-playlist r:\youtube\esrinart --file r:\youtube\esrinart.json
