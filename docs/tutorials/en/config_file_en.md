The path of the configuration file is as follows:

```
config/*
```

### LibContinual Configuration File Composition

LibContinual configuration files use the `yaml` file format. Our predefined configuration files are located in `core/config/default.yaml`, and users can put custom configuration items into the `config/` directory and save them in `.yaml` format.

Although most configurations have been pre-written in `default.yaml`, you cannot directly use the `default.yaml` configuration to run the framework. You need to define the configuration file corresponding to the method you want to run in advance. You can refer to the parameter descriptions below to write your own configuration file.

The `config/headers` folder contains the following files:

- `data.yaml`: Definitions related to data configuration are in this file
- `device.yaml`: Definitions related to GPU configuration items are in this file
- `model.yaml`: Definitions related to model configuration are in this file
- `optimizer.yaml`: Definitions related to the optimizer configuration are in this file

### LibContinual Configuration Settings

#### Data Settings

- `data_root`: The storage path of the dataset
- `image_size`: The size of the input image
- `pin_memory`: Whether to use memory to speed up reading
- `workers`: The number of processes for parallel data reading

```yaml
data_root: /data/cifar10/
image_size: 32
```

#### Model Settings

`backbone`: Backbone network information used in this method

- `name`: The name of the backbone network, which needs to correspond with the implementation in the LibContinual framework

- `kwargs`: Parameters required by the backbone network, which need to be consistent with the naming in the code

  - `num_classes`: The total number of classes needed to be classified by the model
  - `args`: Other required parameters
    - `dataset`: The dataset being used, as different datasets have different backbone network implementation details

  ```yaml
  backbone:
    name: resnet18
    kwargs:
      num_classes: 10
      args: 
        dataset: cifar10
  ```

`classifier`: Classifier information used in the method

- `name`: The name of the classifier, which needs to be consistent with the method implementation in LibContinual

- `kwargs`: Initialization parameters of the classifier, which need to be consistent with the names in the code implementation

  ```yaml
  classifier:
    name: PASS
    kwargs:
      num_class: 100
      feat_dim: 512
      # The following are method-related hyperparameters
      feat_KD: 10.0
      proto_aug: 10.0
      temp: 0.1
  ```

#### Training Settings

- `init_cls_num`: The number of training classes for the first task
- `inc_cls_num`: The number of training classes for subsequent incremental tasks
- `task_num`: The total number of tasks
- `init_epoch`: The number of training epochs for the first task
- `epoch`: The number of training epochs for incremental tasks
- `val_per_epoch`: How many epochs to test performance on the test set
- `batch_size`: Batch size during training
- `warm_up`: The number of warm-up epochs before training

```yaml
warmup: 0
init_cls_num: 50
inc_cls_num: 10
task_num: 6
batch_size: 64
init_epoch: 100
epoch: 100
val_per_epoch: 10
```

#### Optimizer Settings

- `optimizer`: Information about the optimizer used in training
  - `name`: The name of the optimizer, only supports optimizers built into `Pytorch`
  - `kwargs`: Parameters used by this optimizer, parameter names need to match the parameter names in Pytorch optimizers, for example
    - `lr`: Learning rate of the optimizer
    - `weight_decay`: Weight decay

```yaml
optimizer:
  name: Adam
  kwargs:
    lr: 0.001
    weight_decay: 0.0002
```

`lr_scheduler`: Learning rate adjustment strategy used in training, only supports adjustment strategies built into `Pytorch`

- `name`: The name of the learning rate adjustment strategy
- `kwargs`: Parameters of the learning rate adjustment strategy, note that different learning rate adjustment strategies will have different parameters

```yaml
lr_scheduler:
  name: StepLR
  kwargs:
    step_size: 45
    gamma: 0.1
```

#### Hardware Settings

- `device_ids`: GPU IDs used
- `n_gpu`: The number of GPUs used in parallel during training, if it is `1`, it means parallel training is not used
- `deterministic`: Whether to enable `torch.backend.cudnn.benchmark`

```yaml
device_ids: 3
n_gpu: 1
seed: 0
deterministic: False
```

