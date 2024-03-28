# LwF

## Abstract

When building a unified vision system or gradually adding new apabilities to a system, the usual assumption is that training data for all tasks is always available. However, as the number of tasks grows, storing and retraining on such data becomes infeasible. A new problem arises where we add new capabilities to a Convolutional Neural Network (CNN), but the training data for its existing capabilities are unavailable. We propose our Learning without Forgetting method, which uses only new task data to train the network while preserving the original capabilities. Our method performs favorably compared to commonly used feature extraction and fine-tuning adaption techniques and performs similarly to multitask learning that uses original task data we assume unavailable. A more surprising observation is that Learning without Forgetting may be able to replace fine-tuning with similar old and new task datasets for improved new task performance.

![LwF](../../resources/imgs/lwf.gif)

## Citation

```
@article{Li2018LwF,
  author       = {Zhizhong Li and Derek Hoiem},
  title        = {Learning without Forgetting},
  journal      = {{IEEE} Trans. Pattern Anal. Mach. Intell.},
  volume       = {40},
  number       = {12},
  pages        = {2935--2947},
  year         = {2018},
}
```

## How to reproduce LWF

Change the yaml file in `run_trainer.py` as `./config/lwf.yaml`, and then run the code:
```
python run_trainer.py
```

## Results and models

| Backbone | Pretrained |  Dataset  | Epochs |    Split    | Precision |
| :------: | :--------: | :-------: | :----: | :---------: | :-------: |
| Resnet18 |   False    | CIFAR-100 |  100   | Base0 Inc10 |  43.00%   |


