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


## Contributing

We welcome contributions from the community! Whether it's adding new features, improving documentation, or reporting bugs, please feel free to make a pull request or open an issue.

## License

ASRTK is released under the MIT license. Contributions must adhere to this license.