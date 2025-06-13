# Add a new method

下面以[`LUCIR`](https://openaccess.thecvf.com/content_CVPR_2019/html/Hou_Learning_a_Unified_Classifier_Incrementally_via_Rebalancing_CVPR_2019_paper.html)方法为例，描述如何添加一种新的方
法。 <br>

首先，所有方法都继承同一父类`Finetune`。

```python
class Finetune(nn.Module):
    def __init__(self, backbone, feat_dim, num_class, **kwargs):
        ...
        self.kwargs = kwargs
    
    def observe(self, data):
        ...
        return pred, acc / x.size(0), loss

    def inference(self, data):
        ...
        return pred, acc / x.size(0)

    def forward(self, x):
        ...

    def before_task(self, task_idx, buffer, train_loader, test_loaders):
        pass

    def after_task(self, task_idx, buffer, train_loader, test_loaders):
        pass
    
    def get_parameters(self, config):
        ...
        return train_parameters
```
`Finetune`类包含了一个方法需要具备的几个重要接口：
+ `__init__`：初始化函数，用于初始化各方法需要的参数。
+ `observe`：用于训练阶段调用，输入一个batch的训练样本，返回预测、准确率以及前向损失。
+ `inference`：用于推理阶段调用，输入一个batch的样本，返回分类输出、准确率。
+ `forward`：重写`pytorch`的`Module`中的`forward`函数，返回`backbone`的输出。
+ `before_task`：在每个任务开始训练前调用，用于对模型结构、训练参数等进行调整，需要用户自定义。
+ `after_task`：在每个任务开始训练后调用，用于对模型结构、训练参数等进行调整，需要用户自定义。
+ `get_parameters`：在每个任务开始训练前调用，返回当前任务的训练参数。


## LUCIR

### 建立模型
首先在`core/model/replay`下添加`lucir.py`文件：（此处省略部分源码）
```python
class LUCIR(Finetune):
    def __init__(self, backbone, feat_dim, num_class, **kwargs):
        super().__init__(backbone, feat_dim, num_class, **kwargs)
        self.kwargs = kwargs
        self.K = kwargs['K']
        self.lw_mr = kwargs['lw_mr']
        self.ref_model = None


    def before_task(self, task_idx, buffer, train_loader, test_loaders):
        self.task_idx = task_idx

        self.ref_model = copy.deepcopy(self.backbone)
        ...
        new_fc = SplitCosineLinear(in_features, out_features, self.kwargs['inc_cls_num'])

        self.loss_fn1 = nn.CosineEmbeddingLoss()
        self.loss_fn2 = nn.CrossEntropyLoss()
        self.loss_fn3 = nn.MarginRankingLoss(margin=self.kwargs['dist'])
        ...

        self.backbone = self.backbone.to(self.device)
        if self.ref_model is not None:
            self.ref_model = self.ref_model.to(self.device)


    def _init_new_fc(self, task_idx, buffer, train_loader):
        if task_idx == 0:
            return
        ...
        self.backbone.fc.fc2.weight.data = novel_embedding.to(self.device)

    def _compute_feature(self, feature_model, loader, num_samples, num_features):
        ...


    def observe(self, data):
        x, y = data['image'], data['label']
        logit = self.backbone(x)

        ...
        ref_outputs = self.ref_model(x)
        loss = self.loss_fn1(...) * self.cur_lamda
        loss += self.loss_fn2(...)
        if  hard_num > 0:
            ...
            loss += self.loss_fn3(...) * self.lw_mr

        pred = torch.argmax(logit, dim=1)

        acc = torch.sum(pred == y).item()
        return pred, acc / x.size(0), loss

    def after_task(self, task_idx, buffer, train_loader, test_loaders):
        if self.task_idx > 0:
            self.handle_ref_features.remove()
            ...


    def inference(self, data):
        pass


    def _init_optim(self, config, task_idx):
        ...
        tg_params =[{'params': base_params, 'lr': 0.1, 'weight_decay': 5e-4}, \
                        {'params': self.backbone.fc.fc1.parameters(), 'lr': 0, 'weight_decay': 0}]
        return tg_params
```
+ 在`__init__`中，对`LUCIR`所需要的参数`K, lw_mr, ref_model`进行初始化。
+ 在`before_task`中，根据`LUCIR`的需要，我们在任务开始前对新旧分类头进行更新，并根据`task_idx`设置不同的损失函数 。
+ 在`observe`中，我们实现了训练阶段中`LUCIR`的训练算法，根据`task_idx`采用不同的训练方法对模型进行训练。
+ 在`after_task`中，根据`LUCIR`算法需要移除一些`hook`操作。
+ 在`_init_optim`中，我们完成了对于训练参数的选择。

以上几个接口的实现是`LUCIR`算法与其他算法的不同点，其他接口无特殊处理可以不实现交由`Finetune`实现<br>
注意，由于持续学习算法对于第一个任务和其他任务有不同的操作，在`before_task`会传入`task_idx`来标识当前是第几个任务。  <br>




## 新增lucir.yaml文件
各参数含义请参考['config.md'](./config_file_zh.md)
### 数据划分相关参数
```yaml
data_root: /data/fanzhichen/continual/cifar100
image_size: 32
save_path: ./
init_cls_num: 50
inc_cls_num: 10
task_num: 6
```

### 训练优化器相关参数
```yaml
optimizer:
  name: SGD
  kwargs:
    lr: 0.1
    momentum: 0.9
    weight_decay: 0.0005

lr_scheduler:
  name: MultiStepLR
  kwargs:
    gamma: 0.1
    milestones: [80, 120]
```

### backbone相关参数
```yaml
backbone:
  name: resnet32
  kwargs:
    num_classes: 100
    args: 
      dataset: cifar100
      cosine_fc: True
```

### buffer相关参数
`name`: 选择`LinearBuffer`, 会将数据在任务开始前与当前任务数据合并在一起。  <br>
`strategy`：选择`herding`更新策略，目前可支持`random`,`equal_random`,`reservoir`,`herding`,`None` <br>
```yaml
buffer:
  name: LinearBuffer
  kwargs:
    buffer_size: 2000
    batch_size: 128
    strategy: herding     # random, equal_random, reservoir, herding
```


### 算法相关参数
`name`：此处标识所采用何种算法
```yaml
classifier:
  name: LUCIR
  kwargs:
    num_class: 100
    feat_dim: 512
    init_cls_num: 50
    inc_cls_num: 10
    dist: 0.5
    lamda: 5
    K: 2
    lw_mr: 1

```
