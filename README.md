# LibContinual
Make continual learning easy.

## Introduction
LibContinual is an open source continual learning toolbox based on PyTorch. The master branch works with PyTorch 1.13. The compatibility to earlier versions of PyTorch is not fully tested.

![flowchart](./resources/imgs/flowchart.png)


## Supported Methods
### Conventional methods
+ [LwF (ECCV 2016)](https://github.com/RL-VIG/LibContinual/blob/master/reproduce/lwf/README.md)
+ [EWC (PNAS 2017)](https://github.com/RL-VIG/LibContinual/blob/master/reproduce/ewc/README.md)
+ [iCaRL (CVPR 2017)](https://github.com/RL-VIG/LibContinual/blob/master/reproduce/icarl/README.md)
+ [BiC (CVPR 2019)](https://github.com/RL-VIG/LibContinual/blob/master/reproduce/bic/README.md)
+ [LUCIR (CVPR 2019)](https://github.com/RL-VIG/LibContinual/blob/master/reproduce/lucir/README.md)
+ [WA (CVPR 2020)](https://github.com/RL-VIG/LibContinual/blob/master/reproduce/wa/README.md)
+ [DER (CVPR 2021)](https://github.com/RL-VIG/LibContinual/blob/master/reproduce/der/README.md)
+ [OCM (ICML 2022)](https://github.com/RL-VIG/LibContinual/blob/master/reproduce/ocm/README.md)
+ [ERACE,ERAML (ICLR 2022)](https://github.com/RL-VIG/LibContinual/blob/master/reproduce/erace,eraml/README.md)

### Foundation model based methods
+ [L2P (CVPR 2022)](https://github.com/RL-VIG/LibContinual/blob/master/reproduce/l2p/README.md)
+ [DualPrompt (ECCV 2022)](https://github.com/RL-VIG/LibContinual/blob/master/reproduce/dualprompt/README.md)
+ [CodaPrompt (CVPR 2023)](https://github.com/RL-VIG/LibContinual/blob/master/reproduce/codaprompt/README.md)

## Quick Installation

Please refer to [`install.md`](https://libcontinual.readthedocs.io/en/latest/docs/install.html) <br>
Complete tutorials can be found at [`./docs`](https://libcontinual.readthedocs.io/en/latest/)

## Datasets
[CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) are available at [Google Drive](https://drive.google.com/drive/folders/1EL46LQ3ww-F1NVTwFDPIg-nO198cUqWm?usp=sharing)  <br>

After the dataset is downloaded, please extract the compressed file to the specified path.
```
unzip cifar100.zip -d /path/to/your/dataset
```
Set the `data_root` in `.yaml`：
```
data_root: /path/to/your/dataset
```
To add a custom dataset, please refer to [`dataset.md`](https://libcontinual.readthedocs.io/en/latest/docs/data_module_en.html).


## Getting Started

Once you have completed the "Quick Installation" and "Datasets" sections, we can now proceed to demonstrate how to use the "LibContinual" framework with the [`LUCIR`](https://github.com/RL-VIG/LibContinual/blob/master/reproduce/lucir/README.md) method. 

- **Step1:** Set the path in `run_trainer.py` with `./config/lucir.yaml`
    ```python
    config = Config("./config/lucir.yaml").get_config_dict()
    ```
- **Step2:** Configure the parameters in the `./config/lucir.yaml` file. Please refer to [`config.md`](https://libcontinual.readthedocs.io/en/latest/docs/config_file_en.html) for the meanings of each parameter.
- **Step3:** Run code `python run_trainer.py`
- **Step4:** After the training is completed, the log files will be saved in the path specified by the `save_path` parameter.


## Acknowledgement
LibContinual is an open source project designed to help continual learning researchers quickly understand the classic methods and code structures. We welcome other contributors to use this framework to implement their own or other impressive methods and add them to LibContinual. This library can only be used for academic research. We welcome any feedback during using LibContinual and will try our best to continually improve the library.

This project has been developed with references to the following repositories:

- [FACIL](https://github.com/mmasana/FACIL)
- [PyCIL](https://github.com/G-U-N/PyCIL)


We have referenced useful modules from these repositories in our work. We deeply appreciate the authors of these repositories.

## License
This project is licensed under the MIT License. See LICENSE for more details.
