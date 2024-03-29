

配置文件的路径如下：

````
config/*
````

### LibContinual配置文件构成

LibContinual配置文件使用`yaml`文件格式。我们预定义的配置文件位于`core/config/default.yaml`，用户可以将自定义的配置项放入`config/`目录下，并且保存为`.yaml`格式。

虽然大多数配置已经在`default.yaml`提前编写好了，但是您不能直接使用`default.yaml`配置来运行框架，需要预先定义所运行方法对应的配置文件。可以参考下面的参数说明编写你自己的配置文件。

在`config/headers`文件夹中，包含了以下文件：

- `data.yaml`：数据相关的配置定义在此文件中
- `device.yaml`：与GPU相关的配置项定义在此文件中
- `model.yaml`：与模型相关的配置定义此文件中
- `optimizer.yaml`：与优化器相关的配置定义在此文件中

### LibContinual配置文件的设置

#### 数据设置

- `data_root`：数据集的存储路径
- `image_size`：输入图片的大小
- `pin_momery`：是否使用内存来加速读取
- `workers`：并行读取数据进程的数量

```yaml
data_root: /data/cifar10/
image_size: 32
```

#### 模型设置

`backbone`：该方法中使用的骨干网络信息

- `name`: 骨干网络的名称，需要与LibContinual框架中的实现所对应

- `kwargs`：骨干网络所需要的参数，需要与代码中的命名一致

  - `num_classes`：模型需要的分类总数
  - `args`：需要的其他参数
    - `dataset`：所使用的数据集，不同数据集的骨干网络实现细节有所不同

  ```yaml
  backbone:
    name: resnet18
    kwargs:
      num_classes: 10
      args: 
        dataset: cifar10
  ```

`classifier`：方法中使用的分类器信息

- `name`：分类器的名称，需要与LibContinual中的方法实现保持一致

- `kwargs`：分类器的初始化参数，需要与代码实现的名称保持一致

  ```yaml
  classifier:
    name: PASS
    kwargs:
      num_class: 100
      feat_dim: 512
      # 下面是方法相关的超参数
      feat_KD: 10.0
      proto_aug: 10.0
      temp : 0.1
  ```

#### 训练设置

- `init_cls_num`：第一个任务的训练类别数
- `inc_cls_num`：随后增量任务的训练类别数
- `task_num`：任务总数
- `init_epoch`：第一个任务上的训练轮数
- `epoch`：增量任务上的训练轮数
- `val_per_epoch`：每过多少轮训练在测试集上测试性能
- `batch_size`：训练时的批次大小
- `warm_up`：训练之前的预热轮次

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

#### 优化器设置

- `optimizer`：训练中使用的优化器信息
  - `name`：优化器的名称，只支持`Pytorch`内置的优化器
  - `kwargs`：该优化器使用的参数，参数名称需要与Pytorch中优化器参数的参数名称相同，例如
    - `lr`：优化器学习率
    - `weight_decay`：权重衰减

```yaml
optimizer:
  name: Adam
  kwargs:
    lr: 0.001
    weight_decay: 0.0002
```

`lr_scheduler`：训练中使用的学习率调整策略，只支持`Pytorch`内置的优化器调整策略

- `name`：学习率调整策略的名称
- `kwargs`：学习率调整策略的参数，注意不同的学习率调整策略会有不同的参数

```yaml
lr_scheduler:
  name: StepLR
  kwargs:
    step_size: 45
    gamma: 0.1
```

#### 硬件设置

- `device_ids`：所使用的GPU编号
- `n_gpu`：训练中使用的并行GPU数量, 如果是`1`, 表示不使用并行训练
- `deterministic`：是否开启 `torch.backend.cudnn.benchmark` 和 `torch.backend.cudnn.deterministic` 
- `seed`：在 `numpy`，`torch`和 `cuda`中使用的随机种子

```yaml
device_ids: 3
n_gpu: 1
seed: 0
deterministic: False
```
