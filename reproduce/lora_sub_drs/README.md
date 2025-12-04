# LoRA Subtraction for Drift-Resistant Space in Exemplar-Free Continual Learning [(CVPR'2025)](https://openaccess.thecvf.com//content/CVPR2025/papers/Liu_LoRA_Subtraction_for_Drift-Resistant_Space_in_Exemplar-Free_Continual_Learning_CVPR_2025_paper.pdf)

## Abstract
In continual learning (CL), catastrophic forgetting often arises due to feature drift. This challenge is particularly prominent in the exemplar-free continual learning (EFCL) setting, where samples from previous tasks cannot be retained, making it difficult to preserve prior knowledge. To address this issue, some EFCL methods aim to identify feature spaces that minimize the impact on previous tasks while accommodating new ones. However, they rely on static features or outdated statistics stored from old tasks, which prevents them from capturing the dynamic evolution of the feature space in CL, leading to performance degradation over time. In this paper, we introduce the Drift-Resistant Space (DRS), which effectively handles feature drifts without requiring explicit feature modeling or the storage of previous tasks. A novel parameter-efficient fine-tuning approach called Low-Rank Adaptation Subtraction (LoRA ) is proposed to develop the DRS. This method subtracts the LoRA weights of old tasks from the initial pre-trained weight before processing new task data to establish the DRS for model training. Therefore, LoRA enhances stability, improves efficiency, and simplifies implementation. Furthermore, stabilizing feature drifts allows for better plasticity by learning with a triplet loss. Our method consistently achieves state-of-the-art results, especially for long task sequences, across multiple datasets.

![LoRA_SUB_DRS](https://github.com/scarlet0703/LoRA-Sub-DRS/raw/master/imgs/pipeline.png)
s
## Citation

```bibtex
@inproceedings{liu2025lora,
  title={LoRA Subtraction for Drift-Resistant Space in Exemplar-Free Continual Learning},
  author={Liu, Xuan and Chang, Xiaobin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}
```

## How to Reproduce LoRA_SUB_DRS

- **Run command**
    ```python
    python run_trainer.py --config lora_sub-cifar100-b10-10-10
    python run_trainer.py --config lora_sub-cifar100-b5-5-20
    python run_trainer.py --config lora_sub-imgnr-b20-20-10
    python run_trainer.py --config lora_sub-imgnr-b10-10-20
    ```

## Results

### Last Accuracy
|  Dataset   | Backbone | Num of tasks | Buffer size | Reproduced Accuracy | Reported Accuracy |
|  :------:  | :------: | :----------: | :---------: | :-----------------: | :---------------: |
|  CIFAR100  |  SiNet   |      10      |      0      |        89.50        |       89.14       |
|  CIFAR100  |  SiNet   |      20      |      0      |        88.29        |       88.69       |
| Imagenet-R |  SiNet   |      10      |      0      |        75.05        |       74.74       |
| Imagenet-R |  SiNet   |      20      |      0      |        73.62        |       74.80       |

### Overall Average Accuracy
|  Dataset   | Backbone | Num of tasks | Buffer size | Reproduced Accuracy | Reported Accuracy |
|  :------:  | :------: | :----------: | :---------: | :-----------------: | :---------------: |
|  CIFAR100  |  SiNet   |      10      |      0      |        92.55        |       93.11       |
|  CIFAR100  |  SiNet   |      20      |      0      |        92.25        |       92.71       |
| Imagenet-R |  SiNet   |      10      |      0      |        81.16        |       80.35       |
| Imagenet-R |  SiNet   |      20      |      0      |        80.69        |       81.21       |

