# Large Scale Incremental Learning

## Abstract

Modern machine learning suffers from catastrophic forgetting when learning new classes incrementally. The performance dramatically degrades due to the missing data of old classes. Incremental learning methods have been proposed to retain the knowledge acquired from the old classes, by using knowledge distilling and keeping a few exemplars from the old classes. However, these methods struggle to scale up to a large number of classes. We believe this is because of the combination of two factors: (a) the data imbalance between the old and new classes, and (b) the increasing number of visually similar classes. Distinguishing between an increasing number of visually similar classes is particularly challenging, when the training data is unbalanced. We propose a simple and effective method to address this data imbalance issue. We found that the last fully connected layer has a strong bias towards the new classes, and this bias can be corrected by a linear model. With two bias parameters, our method performs remarkably well on two large datasets: ImageNet (1000 classes) and MS-Celeb1M (10000 classes), outperforming the state-of-the-art algorithms by 11.1% and 13.2% respectively.

## Citation
```
@inproceedings{wu2019large,
  title={Large scale incremental learning},
  author={Wu, Yue and Chen, Yinpeng and Wang, Lijuan and Ye, Yuancheng and Liu, Zicheng and Guo, Yandong and Fu, Yun},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={374--382},
  year={2019}
}
```

## How to reproduce Bic

- **Step1: 修改`run_trainer.py`中`Config`参数为`./config/bic.yaml`**

- **Step2: python run_trainer.py**:


## 注意事项

1、对`trainer.py`的修改：bic有两个训练阶段，为了实现二阶段的训练，`trainer.py`部分加了判断条件来判断是否执行第二个训练阶段（不会影响到其它算法跑通），此外，bic需要对buffer和新数据做split，所以`trainer.py`中也在数据处理的部分加了一些判断条件。

2、任务说明：目前仅能做CIFAR100-5这一任务，若需做其它任务，除了修改超参数之外，还需要修改bias_layers部分以及验证集加载部分。



## 复现精度

|                     | 20   | 40    | 60    | 80    | 100   |
| ------------------- | ---- | ----- | ----- | ----- | ----- |
| 原文                | 0.84 | 0.747 | 0.679 | 0.613 | 0.567 |
| Ours(after stage 1) | 0.89 | 0.705 | 0.650 | 0.578 | 0.514 |
| Ours(after stage 2) | 0.89 | 0.725 | 0.687 | 0.628 | 0.578 |