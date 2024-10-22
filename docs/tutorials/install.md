## Installation

This section provides a tutorial on building a working environment for `LibContinual` from scratch.

## Get the `LibContinual` library

Use the following command to get `LibContinual`:

```shell
cd ~
git clone https://github.com/RL-VIG/LibContinual.git
```

## Configure the `LibContinual` environment

The environment can be configured in any of the following ways:

1. conda(recommend)
    ```shell
    cd <path-to-LibContinual> # cd in `LibContinual` directory
    conda env create -f requirements.yaml
    ```

2. pip
    ```shell
    cd <path-to-LibContinual> # cd in `LibContinual` directory
    pip install -r requirements.txt
    ```
3. or whatever works for you as long as the following package version conditions are meet:
    ```
    diffdist==0.1
    numpy==1.21.5
    pandas==1.1.5
    Pillow==9.2.0
    PyYAML==6.0.1
    scikit_learn==1.0.2
    torch==1.12.1
    torchvision==0.13.1
    tqdm==4.64.1
    python==3.8.0
    timm=0.6.7
    ```

## Test the installation


1. set the `config` as follows in `run_trainer.py`:
    ```python
    config = Config("./config/lucir.yaml").get_config_dict()
    ```
2. modify `data_root` in `config/lucir.yaml` to the path of the dataset to be used.
3. run code
   ```shell
   python run_trainer.py
   ```
4. If the first output is correct, it means that `LibContinual` has been successfully installed.

## Next

