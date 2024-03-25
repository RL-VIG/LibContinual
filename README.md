# LibContinual
Make continual learning easy.

## Supported Methods
+ [BiC (CVPR 2019)](https://arxiv.org/abs/1905.13260)
+ [EWC (PNAS 2017)](https://arxiv.org/abs/1612.00796)
+ [iCaRL (CVPR2017)](https://arxiv.org/abs/1611.07725)
+ [LUCIR (CVPR 2019)](https://openaccess.thecvf.com/content_CVPR_2019/html/Hou_Learning_a_Unified_Classifier_Incrementally_via_Rebalancing_CVPR_2019_paper.html)
+ [LwF (ECCV 2016)](https://arxiv.org/abs/1606.09282)
+ [WA (CVPR 2020)](https://arxiv.org/abs/1911.07053)
+ [OCM (PMLR 2022)](https://proceedings.mlr.press/v162/guo22g.html)
+ [DER (CVPR 2021)](https://arxiv.org/abs/2103.16788)


## Quick Installation(tod)
(待文档部分完成)  <br>
请参考文档中[`安装`](https://github.com/RL-VIG/LibContinual/blob/master/docs/tutorials/install.md)部分。 <br>
完整文档：[`./docs`](https://github.com/RL-VIG/LibContinual/tree/master/docs)

## Datasets(todo)
[`CIFAR-100`](https://drive.google.com/drive/folders/1EL46LQ3ww-F1NVTwFDPIg-nO198cUqWm?usp=sharing), `miniImageNet(todo)`  <br>

将对应数据集的压缩包解压至指定路径:
```
unzip cifar100.zip -d /path/to/your/dataset
```
修改.yaml文件的data_root参数：
```
data_root: /path/to/your/dataset
```
如何添加自定义数据集请参考文档:[`添加自定义数据集`](https://github.com/RL-VIG/LibContinual/blob/master/docs/tutorials/data_module.md)


## Acknowledgement
LibContinual is an open source project designed to help continual learning researchers quickly understand the classic methods and code structures. We welcome other contributors to use this framework to implement their own or other impressive methods and add them to LibContinual. This library can only be used for academic research. We welcome any feedback during using LibContinual and will try our best to continually improve the library.

## License
This project is licensed under the MIT License. See LICENSE for more details.