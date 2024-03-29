# Write a `.yaml` configuration file

Code for this section:
```
core/config/config.py
config/*
```

## Composition of the configuration file in LibContinual

The configuration file of  LibContinual uses a yaml format file and it also supports reading the global configuration changes from the command line. We have pre-defined a default configuration `core/config/default.yaml`. The users can put the custom configuration into the `config/` directory, and save this file in the `yaml` format. At parsing, the sequencing relationship of defining the configuration of the method is `default.yaml->config/->console`. The latter definition overrides the same value in the former definition.

Although most of the basic configurations have been set in the `default.yaml`, you can not directly run a program just using the `default.yaml`. Before running the code, the users are required to define a configuration file of one method that has been implemented in  LibContinual in the `config/` directory.

Considering that CL menthods usually have some basic parameters, such as `image_size` or `device id`, which are often needed to be changed,  LibContinual also supports making changes to some simple configurations on the command line without modifying the `yaml` file. Similarly, during training and test, because many parameters are the same of different methods, we wrap these same parameters together and put them into the`config/headers` for brevity. In this way, we can write the `yaml` files of the custom methods succinctly by importing them.

The following is the composition of the files in the `config/headers` directory.

- `data.yaml`: The relevant configuration of the data is defined in this file.
- `device.yaml`: The relevant configuration of GPU is defined in this file.
- `model.yaml`: The relevant configuration of the model is defined in this file.
- `optimizer.yaml`: The relevant configuration of the optimizer used for training is defined in this file.

## The settings of the configuration file in  LibContinual

The following details each part of the configuration file and explain how to write them. An example of how the bic method is configured is also presented.

### The settings for data

+ `data_root`: The storage path of the dataset.

+ `image_size`: The size of the input image.

+ `pin_memory`: Whether to use memory acceleration for reading.

+ `augment`: Whether to use data augmentation.

+ `init_cls_num`: Initial number of classes.

+ `inc_cls_num`: Incremental number of classes.

+ `task_num`: Number of tasks.

+ `works`: Number of working threads for data loading and preprocessing.

  ```yaml
  data_root: /data/cifar100
  image_size: 84
  pin_memory: False
  augment: True
  init_cls_num: 20
  inc_cls_num: 20
  task_num: 5
  works: 8 
  ```

### The settings for model

+ `backbone`: The `backbone` information used in the method.
  + `name`: The name of the `backbone`, needs to match the case of the `backbone` implemented in  LibContinual.
  + `kwargs`: The parameters used in the `backbone`, must keep the name consistent with the name in the code.
    + `num_classes`: Number of classes.
    + `args`: Other parameters, for example, the dataset used.
  
```yaml
  backbone:
  	name: resnet18
      kwargs:
      	num_classes: 100
        	args: 
        		dataset: cifar100
```
  
+ `classifier`: The `classifier` information used in the method.
  + `name`: The name of the `classifier`, needs to match the case of the `classifier` implemented in  LibContinual.
  
+ `kwargs`: The parameters used in the `classifier` initialization, must keep the name consistent with the name in the code.

  + `feat_dim`: Dimension settings

  ```yaml
  classifier:
      name: bic
      kwargs:
      	feat_dim: 512
  ```

### The settings for training

+ `epoch`: The number of `epoch` during training.

+ `test_epoch`: The number of `epoch` during testing.

+ `val_per_epoch`: The number of `epoch` in each verification phase.

+ `stage2_epoch`:  The number of `epoch` for strategy 2.

+ `batch_size`:  The batch size for training.

  ```yaml
  epoch: 50
  test_epoch: 5
  val_per_epoch: 5
  stage2_epoch: 100
  batch_size: 128
  ```

### The settings for optimizer

+ `optimizer`: Optimizer information used during training.

  + `name`: The name of the Optimizer, only temporarily supports all Optimizers provided by `PyTorch`.
  + `kwargs`: The parameters used in the optimizer, and the name needs to be the same as the parameter name required by the `PyTorch` optimizer.
  + `other`: Currently, the framework only supports the learning rate used by each part of a separately specified method, and the name needs to be the same as the variable name used in the method.

  ```yaml
  optimizer:
      name: SGD
      kwargs:
          lr: 0.01
          weight_decay: 2e-4
          momentum: 0.9
  ```

+ `lr_scheduler`: The learning rate adjustment strategy used during training, only temporarily supports all the learning rate adjustment strategies provided by `PyTorch`.
  + `name`: The name of the learning rate adjustment strategy.
  + `kwargs`: Other parameters used in the learning rate adjustment strategy in `PyTorch`.
  
  ```yaml
  lr_scheduler:
    name: MultiStepLR
    kwargs:
      gamma: 0.1
      milestones: [25, 50]
  ```

### The settings for Hardware

+ `device_ids`: The `gpu` number, which is the same as the `nvidia-smi` command.

+ `n_gpu`: The number of parallel `gpu` used during training, if `1`, it can't apply to parallel training.

+ `seed`: Seed points used in `numpy`，`torch`，and `cuda`.

+ `deterministic`: Whether to turn on `torch.backend.cudnn.benchmark` and `torch.backend.cudnn.deterministic` and whether to determine random seeds during training.

  ```yaml
  device_ids: 0,1,2,3,4,5,6,7
  n_gpu: 4
  seed: 1993
  deterministic: False
  ```

  
