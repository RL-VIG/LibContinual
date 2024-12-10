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

- **Step 1 : Configure `./config/erace.yaml` and `./config/eraml.yaml`**

- **Step 2 : Run command**
    ```python
    python run_trainter.py --config_name erace
    python run_trainter.py --config_name eraml
    ```

## Results

### ERACE

| Dataset  | Num of Tasks | Buffer Size | Epochs | Reproduced Accuracy | Reported Accuray |
| :------: | :----------: | :---------: | :----: | :-----------------: | :--------------: |
| CIFAR10  |      5       |    20*10    |   1    |        31.4         |       42.82      |
| CIFAR10  |      5       |    20*10    |   5    |        42.2         |       49.40      |
| CIFAR10  |      5       |    20*10    |   15   |        46.6         |       44.92      |

| Dataset  | Num of Tasks | Buffer Size | Epochs | Reproduced Accuracy | Reported Accuray |
| :------: | :----------: | :---------: | :----: | :-----------------: | :--------------: |
| CIFAR100 |      20      |    20*100   |   1    |       12.10         |      17.46       |
| CIFAR100 |      20      |    20*100   |   5    |       21.20         |      18.26       |
| CIFAR100 |      20      |    20*100   |   15   |       32.20         |      15.78       |

### ERAML

| Dataset  | Num of Tasks | Buffer Size | Epochs | Reproduced Accuracy | Reported Accuray |
| :------: | :----------: | :---------: | :----: | :-----------------: | :--------------: |
| CIFAR10  |      5       |    20*10    |   1    |        29.6         |       37.48      |
| CIFAR10  |      5       |    20*10    |   5    |        41.2         |       39.92      |
| CIFAR10  |      5       |    20*10    |   15   |        44.2         |       35.58      |

| Dataset  | Num of Tasks | Buffer Size | Epochs | Reproduced Accuracy | Reported Accuray |
| :------: | :----------: | :---------: | :----: | :-----------------: | :--------------: |
| CIFAR100 |      20      |    20*100   |   1    |       7.00          |      10.26       |
| CIFAR100 |      20      |    20*100   |   5    |       19.20         |      14.52       |
| CIFAR100 |      20      |    20*100   |   15   |       25.30         |      12.22       |