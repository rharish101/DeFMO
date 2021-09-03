# Improving DeFMO With Learned Losses

This is the repository for my semester project at ETH Zürich under Denys Rozumnyi and Prof. Dr. Marc Pollefeys.
The project extends [DeFMO: Deblurring and Shape Recovery of Fast Moving Objects](https://arxiv.org/abs/2012.00595) using learned losses.

## Description
For a detailed description of this project, check out the project report in [report.pdf](./report.pdf).
For more info about DeFMO, the base for this project, check out the code in the [DeFMO repository](https://github.com/rozumden/DeFMO).

## Instructions

All Python scripts use argparse to parse commandline arguments.
For viewing the list of all positional and optional arguments for any script, type:
```sh
./script.py --help
```

### Setup
[Poetry](https://python-poetry.org/) is used for conveniently installing and managing dependencies.

1. *[Optional]* Create and activate a virtual environment with Python >= 3.7.

2. Install Poetry globally (recommended), or in a virtual environment.
    Please refer to [Poetry's installation guide](https://python-poetry.org/docs/#installation) for recommended installation options.

    You can use pip to install it:
    ```sh
    pip install poetry
    ```

3. Install all dependencies with Poetry:
    ```sh
    poetry install --no-dev
    ```

    If you didn't create and activate a virtual environment in step 1, Poetry creates one for you and installs all dependencies there.
    To use this virtual environment, run:
    ```sh
    poetry shell
    ```

#### For Contributing
[pre-commit](https://pre-commit.com/) is used for managing hooks that run before each commit, to ensure code quality and run some basic tests.
Thus, this needs to be set up only when one intends to commit changes to git.

1. Activate the virtual environment where you installed the dependencies.

2. Install all dependencies, including extra dependencies for development:
    ```sh
    poetry install
    ```

3. Install pre-commit hooks:
    ```sh
    pre-commit install
    ```

**NOTE**: You need to be inside the virtual environment where you installed the above dependencies every time you commit.
However, this is not required if you have installed pre-commit globally.

### Hyper-Parameter Configuration
Hyper-parameters can be specified through [TOML](https://toml.io/en/) configs.
For example, to specify a batch size of 32 and a learning rate of 0.001, use the following config:
```toml
batch_size = 32
lr = 0.001
```

You can store configs in a directory named `configs` located in the root of this repository.
It has an entry in the [`.gitignore`](./.gitignore) file so that custom configs aren't picked up by git.

The available hyper-parameters, their documentation and default values are specified in the `Config` class in the file [`config.py`](./config.py).

### Training
Run `train.py`:
```sh
./train.py /path/to/dataset/ /path/where/to/save/logs/
```

The weights of trained models are saved in PyTorch's `.pt` format inside an ISO 8601 timestamped subdirectory, which is stored in a parent directory.
This parent directory is given by the second positional argument (`/path/where/to/save/logs/` here).

Training logs are saved in the `training` subdirectory of this timestamped directory.
The hyper-parameter config, along with the current date and time, is saved as a TOML file named `config.toml` in this `training` subdirectory.

### Inference
The script `run.py` is provided to run a model on either an input image (with a background) or an input video.
To run it on an image with a background, run:
```sh
./run.py --im /path/to/input/image --bgr /path/to/input/background /path/to/trained/model/weights/
```

To run it on a video, run:
```sh
./run.py --video /path/to/input/video /path/to/trained/model/weights/
```

The outputs are saved in the directory given by the `--output-folder` argument.
By default, this directory is `output`.
