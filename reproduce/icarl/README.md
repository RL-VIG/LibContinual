# iCaRL: Incremental Classifier and Representation Learning [(CVPR'2017)](https://arxiv.org/abs/1611.07725)


## Abstract

A major open problem on the road to artificial intelligence is the development of incrementally learning systems that learn about more and more concepts over time from a stream of data. In this work, we introduce a new training strategy, iCaRL, that allows learning in such a class-incremental way: only the training data for a small number of classes has to be present at the same time and new classes can be added progressively.

iCaRL learns strong classifiers and a data representation simultaneously. This distinguishes it from earlier works that were fundamentally limited to fixed data representations and therefore incompatible with deep learning architectures. We show by experiments on CIFAR-100 and ImageNet ILSVRC 2012 data that iCaRL can learn many classes incrementally over a long period of time where other strategies quickly fail.



# Citation
```
@inproceedings{rebuffi2017icarl,
  title={icarl: Incremental classifier and representation learning},
  author={Rebuffi, Sylvestre-Alvise and Kolesnikov, Alexander and Sperl, Georg and Lampert, Christoph H},
  booktitle={Proceedings of the IEEE conference on Computer Vision and Pattern Recognition},
  pages={2001--2010},
  year={2017}
}
```


## How to reproduce iCaRL


- **Step1: 修改`run_trainer.py`中`Config`参数为`./config/icarl.yaml`**

- **Step2: python run_trainer.py**:


## 复现精度



|  数据集  | 任务数量 | buffer size | 复现准确度 |
| :------: | :------: | :---------: | :--------: |
| CIFAR100 |    5     |    2000     |    54.4    |
| CIFAR100 |    10    |    2000     |    46.5    |



