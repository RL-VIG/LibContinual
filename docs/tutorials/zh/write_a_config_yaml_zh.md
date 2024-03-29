# 编写`.yaml`配置文件

本节相关代码：
```
core/config/config.py
config/*
```

## LibContinual中配置文件的组成

LibContinual的配置文件采用了yaml格式的文件，同时也支持从命令行中读取一些全局配置的更改。我们预先定义了一个默认的配置`core/config/default.yaml`。用户可以将自定义的配置放在`config/`目录下，保存为`yaml`格式的文件。配置定义在解析时的优先级顺序是`default.yaml->config/->console`。后一个定义会覆盖前一个定义中名称相同的值。

尽管`default.yaml`中设置的是持续学习中的一些最基础的配置，无法仅依靠`default.yaml`直接运行程序。运行代码前，用户需要在`config/`目录下定义已经在LibContinual中实现了的方法的配置。

考虑到持续方法有一些基本参数例如`image_sie， epoch`或者`device id`，这样的参数是经常需要改动的。LibContinual支持在命令行中对一些简单的配置进行更改而不需要修改`yaml`文件。同样的，在训练和测试过程中，很多不同的持续学习方法的参数是相同的。为了简洁起见，我们将这些相同的参数包装到了一起，放到了`config/headers`目录下，这样就能够通过导入的方式简洁地编写自定义方法的`yaml`文件。

以下是`config/headers`目录下文件的构成。

- `data.yaml`：定义了训练所使用的数据的相关配置。
- `device.yaml`：定义了训练所使用的GPU的相关配置。
- `model.yaml`：定义了模型训练的相关配置。
- `optimizer.yaml`：定义了训练所使用的优化器的相关配置。

## LibContinual中配置文件的设置

以下详细介绍配置文件中每部分代表的信息以及如何编写，以下将以bic方法的配置给出示例。

### 数据设置

+ `data_root`：数据集存放的路径。

+ `image_size`：输入图像的尺寸。

+ `pin_momery`：是否使用内存加速读取。

+ `augment`：是否使用数据增强。

+ `init_cls_num`:  初始类别数量。

+ `inc_cls_num`:  增量类别数量。

+ `task_num`:  任务数量。

+ `works`：数据加载和预处理的工作线程数量。

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


### 模型设置

+ `backbone`：方法所使用的`backbone`信息。
  
  + `name`：使用的backbone的名称，需要与LibContinual中实现的backbone的大小写一致。
  + `kwargs`：`backbone`初始化时用到的参数，必须保持名称与代码中的名称一致。
    + `num_classes`：类别数量。
    + `args`：其他项参数，例如所使用的数据集`dataset`。
  
  ```yaml
  backbone:
  	name: resnet18
      kwargs:
      	num_classes: 100
        	args: 
        		dataset: cifar100
  ```
+ `classifier`：方法所使用的方法信息。
  
  + `name`：使用的方法的名称，需要与LibContinual中实现的方法的名称一致。
+ `kwargs`：方法初始化时用到的参数，必须保持名称与代码中的名称一致。
    
    + `feat_dim`：维度设定。
  
  ```yaml
  classifier:
      name: bic
      kwargs:
      	feat_dim: 512
  ```

### 训练设置

+ `epoch`：训练的`epoch`数。

+ `test_epoch`: 测试的`epoch`数。

+ `val_per_epoch`:  验证阶段的每一次的`epoch`数。

+ `stage2_epoch`:  策略2的`epoch`数。

+ `batch_size`:  训练的批次尺寸。

  ```yaml
  epoch: 50
  test_epoch: 5
  val_per_epoch: 5
  stage2_epoch: 100
  batch_size: 128
  ```

### 优化器设置

+ `optimizer`：训练阶段使用的优化器信息。
  + `name`：优化器名称，当前仅支持`Pytorch`提供的所有优化器。
  + `kwargs`：传入优化器的参数，名称需要与Pytorch优化器所需要的参数名称相同。
  + `other`：当前仅支持单独指定方法中的每一部分所使用的学习率，名称需要与方法中所使用的变量名相同。

  ```yaml
  optimizer:
      name: SGD
      kwargs:
          lr: 0.01
          weight_decay: 2e-4
          momentum: 0.9
  ```
  
+ `lr_scheduler`：训练时使用的学习率调整策略，当前仅支持`Pytorch`提供的所有学习率调整策略。
  + `name`：学习率调整策略名称。
  + `kwargs`：其他`Pytorch`学习率调整策略所需要的参数。

  ```yaml
  lr_scheduler:
    name: MultiStepLR
    kwargs:
      gamma: 0.1
      milestones: [25, 50]
  ```

### 硬件设置

+ `device_ids`：训练可以用到的`gpu`的编号，与`nvidia-smi`命令显示的编号相同。

+ `n_gpu`：训练使用并行训练的`gpu`个数，如果仅有`1`个GPU的话，则不适用并行训练。

+ `seed`：训练时`numpy`，`torch`，`cuda`使用的种子点。

+ `deterministic`：是否开启`torch.backend.cudnn.benchmark`以及`torch.backend.cudnn.deterministic`以及是否使训练随机种子确定。

  ```yaml
  device_ids: 0,1,2,3,4,5,6,7
  n_gpu: 4
  seed: 1993
  deterministic: False
  ```
