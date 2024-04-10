# ERACE, ERAML: New Insights on reducing abrupt representation change in online continual learning [(ICLR'2022)](https://arxiv.org/abs/2104.05025)

## Abstract

In the online continual learning paradigm, agents must learn from a changing distribution while respecting memory and compute constraints. Experience Replay (ER), where a small subset of past data is stored and replayed alongside new data, has emerged as a simple and effective learning strategy. In this work, we focus on the change in representations of observed data that arises when previously unobserved classes appear in the incoming data stream, and new classes must be distinguished from previous ones. We shed new light on this question by showing that applying ER causes the newly added classes' representations to overlap significantly with the previous classes, leading to highly disruptive parameter updates. 

Based on this empirical analysis, we propose a new method which mitigates this issue by shielding the learned representations from drastic adaptation to accommodate new classes. We show that using an asymmetric update rule pushes new classes to adapt to the older ones (rather than the reverse), which is more effective especially at task boundaries, where much of the forgetting typically occurs.

## Citation
```bibtex
@misc{caccia2022new,
      title={New Insights on Reducing Abrupt Representation Change in Online Continual Learning}, 
      author={Lucas Caccia and Rahaf Aljundi and Nader Asadi and Tinne Tuytelaars and Joelle Pineau and Eugene Belilovsky},
      year={2022},
      eprint={2104.05025},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## How to Reproduce ERACE, ERAML

- **Step1: Set the path in `run_trainer.py` with `./config/erace.yaml` or `./config/eraml.yaml`**
    ```python
    config = Config("./config/erace.yaml").get_config_dict()
    config = Config("./config/eraml.yaml").get_config_dict()
    ```
- **Step2: Run command**
    ```python
    python run_trainer.py
    ```

## Results

| Dataset | Num of Tasks | Buffer Size | Epochs | Reproduced Accuracy |
| :-----: | :----------: | :---------: | :----: | :-----------------: |
| CIFAR10 |      20      |    2000     |   1    |        0.121        |
| CIFAR10 |      20      |    2000     |   5    |        0.231        |
| CIFAR10 |      20      |    2000     |   15   |        0.327        |

| Dataset  | Num of Tasks | Buffer Size | Epochs | Reproduced Accuracy |
| :------: | :----------: | :---------: | :----: | :-----------------: |
| CIFAR100 |      20      |    2000     |   1    |       0.070         |
| CIFAR100 |      20      |    2000     |   5    |       0.206         |
| CIFAR100 |      20      |    2000     |   15   |       0.253         |



