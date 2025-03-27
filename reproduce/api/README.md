# API : Adaptive Plasticity Improvement for Continual Learning [(CVPR'2023)](https://openaccess.thecvf.com/content/CVPR2023/papers/Liang_Adaptive_Plasticity_Improvement_for_Continual_Learning_CVPR_2023_paper.pdf)

## Abstract

Many works have tried to solve the catastrophic forgetting (CF) problem in continual learning (lifelong learning). However, pursuing non-forgetting on old tasks may damage the model’s plasticity for new tasks. Although some methods have been proposed to achieve stability-plasticity trade-off, no methods have considered evaluating a model’s plasticity and improving plasticity adaptively for a new task. In this work, we propose a new method, called adaptive plasticity improvement (API), for continual learning. Besides the ability to overcome CF on old tasks, API also tries to evaluate the model’s plasticity and then adaptively improve the model’s plasticity for learning a new task if necessary. Experiments on several real datasets show that API can outperform other state-of-the-art baselines in terms of both accuracy and memory usage.

## Citation

```bibtex
@inproceedings{
  liang2023adaptive,
  title={Adaptive Plasticity Improvement for Continual Learning},
  author={Liang, Yan-Shuo and Li, Wu-Jun},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={7816--7825},
  year={2023}
}
```

## How to Reproduce GPM

- **Step 1 : Run one of these commands**
    ```python
    python run_trainter.py --config_name api_til-alexnet-cifar100-b5-5-20
    ```

## Results

| Dataset  | Scenario | Num of Tasks | Epochs | Reproduced Accuracy | Reported Accuracy |
| :------: | :------: |:-----------: | :----: | :-----------------: | :---------------: |
| CIFAR100 |   TIL    |      20      |  200   |       81.24         |       81.40       |
