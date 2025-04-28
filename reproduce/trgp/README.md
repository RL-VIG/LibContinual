# TRGP: Trust Region Gradient Projection for Continual Learning [(ICLR 2022)](https://arxiv.org/abs/2202.02931)

## Abstract
Catastrophic forgetting is one of the major challenges in continual learning. To address this issue, some existing methods put restrictive constraints on the optimization space of the new task for minimizing the interference to old tasks. However, this may lead to unsatisfactory performance for the new task, especially when the new task is strongly correlated with old tasks. To tackle this challenge, we propose Trust Region Gradient Projection (TRGP) for continual learning to facilitate the forward knowledge transfer based on an efficient characterization of task correlation. Particularly, we introduce a notion of `trust region' to select the most related old tasks for the new task in a layer-wise and single-shot manner, using the norm of gradient projection onto the subspace spanned by task inputs. Then, a scaled weight projection is proposed to cleverly reuse the frozen weights of the selected old tasks in the trust region through a layer-wise scaling matrix. By jointly optimizing the scaling matrices and the model, where the model is updated along the directions orthogonal to the subspaces of old tasks, TRGP can effectively prompt knowledge transfer without forgetting. Extensive experiments show that our approach achieves significant improvement over related state-of-the-art methods.

## Citation

```bibtex
@article{lin2022trgp,
  title={TRGP: Trust Region Gradient Projection for Continual Learning},
  author={Lin, Sen and Yang, Li and Fan, Deliang and Zhang, Junshan},
  journal={arXiv preprint arXiv:2202.02931},
  year={2022}
}
```

## How to Reproduce

- **Step1: Set the path in `run_trainer.py` with `./config/trgp.yaml`**
  ```python
  config = Config("./config/trgp.yaml").get_config_dict()
  ```
- **Step2: Run command**
  ```python
  python run_trainer.py
  ```

## Results

| Dataset  | Backbone | Num of tasks | Buffer size | Epochs | Reproduced Accuracy |  Reported Accuracy  |
| :------: | :------: | :----------: | :---------: | :----: | :-----------------: | :-----------------: |
| CIFAR100 | AlexNet  |      10      |      0      |   200  |        77.77        |        74.49        |

