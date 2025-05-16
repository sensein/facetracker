# The ```facetracker``` repo

[![Build](https://github.com/sensein/facetracker/actions/workflows/test.yaml/badge.svg?branch=main)](https://github.com/sensein/facetracker/actions/workflows/test.yaml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/sensein/facetracker/branch/main/graph/badge.svg?token=014fb51c-ea46-4f81-83e3-80dac272bef3)](https://codecov.io/gh/sensein/facetracker)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

[![PyPI](https://img.shields.io/pypi/v/facetracker.svg)](https://pypi.org/project/facetracker/)
[![Python Version](https://img.shields.io/pypi/pyversions/facetracker)](https://pypi.org/project/facetracker)
[![License](https://img.shields.io/pypi/l/facetracker)](https://opensource.org/licenses/Apache-2.0)

[![pages](https://img.shields.io/badge/api-docs-blue)](https://sensein.github.io/facetracker)

Welcome to the ```facetracker``` repo! This is a Python package for doing incredible stuff.

**Caution:**: this package is still under development and may change rapidly over the next few weeks.

## Installation

1. Install the base requirements:
   ```bash
   pip install -r requirements-torch.txt
   pip install -r requirements-torch-dev.txt
   ```

2. Install OpenMIM and use it to install mmcv and mmdet:
   ```bash
   pip install -U openmim
   mim install mmcv
   mim install mmdet
   ```

3. Install mmpose from source:
   ```bash
   git clone https://github.com/open-mmlab/mmpose.git
   cd mmpose
   pip install -r requirements.txt
   pip install -v -e .
   ```
   Note: The `-v` flag enables verbose output, and `-e` installs the project in editable mode so that local modifications take effect without reinstallation.

4. Download the required config and checkpoint files:
   ```bash
   mim download mmpose --config td-hm_hrnet-w48_8xb32-210e_coco-256x192 --dest .
   ```

5. For more detailed installation instructions and customization options, refer to the [official MMPose installation guide](https://mmpose.readthedocs.io/en/latest/installation.html).

6. Install SAM2:
   ```bash
   git clone https://github.com/facebookresearch/sam2.git
   cd sam2
   pip install -e .
   ```
   For more details, refer to the [SAM2 GitHub repository](https://github.com/facebookresearch/sam2).

7. For TensorFlow setup, refer to the `facetracker.def` file for a separate installation process to avoid conflicts with PyTorch. TensorFlow is used for face detection with RetinaFace, and we plan to integrate TensorFlow and PyTorch together later.


