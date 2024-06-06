# LibContinual
Make continual learning easy.

## Introduction
LibContinual is an open source continual learning toolbox based on PyTorch. The master branch works with PyTorch 1.13. The compatibility to earlier versions of PyTorch is not fully tested.

![flowchart](./resources/imgs/flowchart.png)

## Supported Methods
+ [BiC (CVPR 2019)](https://github.com/RL-VIG/LibContinual/blob/master/reproduce/bic/README.md)
+ [EWC (PNAS 2017)](https://github.com/RL-VIG/LibContinual/blob/master/reproduce/ewc/README.md)
+ [iCaRL (CVPR2017)](https://github.com/RL-VIG/LibContinual/blob/master/reproduce/icarl/README.md)
+ [LUCIR (CVPR 2019)](https://github.com/RL-VIG/LibContinual/blob/master/reproduce/lucir/README.md)
+ [LwF (ECCV 2016)](https://github.com/RL-VIG/LibContinual/blob/master/reproduce/lwf/README.md)
+ [WA (CVPR 2020)](https://github.com/RL-VIG/LibContinual/blob/master/reproduce/wa/README.md)
+ [OCM (PMLR 2022)](https://github.com/RL-VIG/LibContinual/blob/master/reproduce/ocm/README.md)
+ [DER (CVPR 2021)](https://github.com/RL-VIG/LibContinual/blob/master/reproduce/der/README.md)
+ [ERACE,ERAML (ICLR 2022)](https://github.com/RL-VIG/LibContinual/blob/master/reproduce/erace,eraml/README.md)
+ [L2P (CVPR 2022)](https://github.com/RL-VIG/LibContinual/blob/master/reproduce/l2p/README.md)
+ [DualPrompt (ECCV 2022)](https://github.com/RL-VIG/LibContinual/blob/master/reproduce/dualprompt/README.md)
+ [CodaPrompt (CVPR 2023)](https://github.com/RL-VIG/LibContinual/blob/master/reproduce/codaprompt/README.md)

## Quick Installation
(待文档部分完成)  <br>
请参考文档中[`安装`](https://github.com/RL-VIG/LibContinual/blob/master/docs/tutorials/install.md)部分。 <br>
完整文档：[`./docs`](https://github.com/RL-VIG/LibContinual/tree/master/docs)

## Datasets
[`CIFAR-100`](https://drive.google.com/drive/folders/1EL46LQ3ww-F1NVTwFDPIg-nO198cUqWm?usp=sharing), `miniImageNet(todo)`  <br>

将对应数据集的压缩包解压至指定路径:
```
unzip cifar100.zip -d /path/to/your/dataset
```
修改.yaml文件的data_root参数：
```
data_root: /path/to/your/dataset
```
如何添加自定义数据集请参考文档:[`添加自定义数据集`](https://github.com/RL-VIG/LibContinual/blob/master/docs/tutorials/zh/data_module_zh.md)

## Get Start

当您已经完成`Quick Installation`和`Datasets`后，我们以`LUCIR`方法为例展示如何使用`LibContinual`。
- **Step1**: 修改`run_trainer.py`中`Config`参数为`./config/lucir.yaml`
- **Step2**：配置`./config/lucir.yaml`文件中的参数，各参数含义请参考[配置文件](https://github.com/RL-VIG/LibContinual/blob/master/docs/tutorials/config_file.md)
- **Step3**: 运行代码`python run_trainer.py`
- **Step4**：日志保存在配置文件中`save_path`路径下


## Acknowledgement
LibContinual is an open source project designed to help continual learning researchers quickly understand the classic methods and code structures. We welcome other contributors to use this framework to implement their own or other impressive methods and add them to LibContinual. This library can only be used for academic research. We welcome any feedback during using LibContinual and will try our best to continually improve the library.


在本项目开发过程中参考了下列仓库：

- [FACIL](https://github.com/mmasana/FACIL)
- [PyCIL](https://github.com/G-U-N/PyCIL)

在我们的工作中参考了这些仓库中有用的模块。我们深深感谢这些仓库的作者们。

## License
This project is licensed under the MIT License. See LICENSE for more details.