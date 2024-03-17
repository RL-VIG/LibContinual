# Add a new method

我们以`LUCIR`(Learning_a_Unified_Classifier_Incrementally_via_Rebalancing)方法为例，描述如何添加一种新的方法。 <br>
首先，我们了解一下所有方法的共同父类`Finetune`。

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

以上这些接口也是新增方法所必需要实现的接口。

## LUCIR
接下来以`LUCIR`为例，描述如何在`LibContinual`中新增一个方法

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

        if task_idx == 1:
            self.ref_model = copy.deepcopy(self.backbone)
            in_features = self.backbone.fc.in_features
            out_features = self.backbone.fc.out_features
            new_fc = SplitCosineLinear(in_features, out_features, self.kwargs['inc_cls_num'])
            ...
            self.backbone.fc = new_fc
            lamda_mult = out_features*1.0 / self.kwargs['inc_cls_num']


        elif task_idx > 1:
            self.ref_model = copy.deepcopy(self.backbone) # 应该带上classifier
            in_features = self.backbone.fc.in_features
            out_features1 = self.backbone.fc.fc1.out_features
            out_features2 = self.backbone.fc.fc2.out_features
            new_fc = SplitCosineLinear(in_features, out_features1+out_features2, self.kwargs['inc_cls_num']).to(self.device)
            ...
            self.backbone.fc = new_fc
            lamda_mult = (out_features1+out_features2)*1.0 / (self.kwargs['inc_cls_num'])
        
        if task_idx > 0:
            self.cur_lamda = self.kwargs['lamda'] * math.sqrt(lamda_mult)
        else:
            self.cur_lamda = self.kwargs['lamda']


        self._init_new_fc(task_idx, buffer, train_loader)

        if task_idx == 0:
            self.loss_fn = nn.CrossEntropyLoss()
        else:
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
        old_embedding_norm = self.backbone.fc.fc1.weight.data.norm(dim=1, keepdim=True)   # 旧类向量
        average_old_embedding_norm = torch.mean(old_embedding_norm, dim=0).to('cpu').type(torch.DoubleTensor)   # 旧类向量的均值
        feature_model = nn.Sequential(*list(self.backbone.children())[:-1])
        num_features = self.backbone.fc.in_features
        novel_embedding = torch.zeros((self.kwargs['inc_cls_num'], num_features))

        tmp_datasets = copy.deepcopy(train_loader.dataset)
        ...
        
        self.backbone.to(self.device)
        self.backbone.fc.fc2.weight.data = novel_embedding.to(self.device)

    def _compute_feature(self, feature_model, loader, num_samples, num_features):
        ...
        return features


    def observe(self, data):
        x, y = data['image'], data['label']
        x = x.to(self.device)
        y = y.to(self.device)
        logit = self.backbone(x)

        if self.task_idx == 0:
            loss = self.loss_fn(logit, y)
        else:
            ref_outputs = self.ref_model(x)
            loss = self.loss_fn1(cur_features, ref_features.detach(), \
                    torch.ones(x.size(0)).to(self.device)) * self.cur_lamda
            
            loss += self.loss_fn2(logit, y)
            ...
            if  hard_num > 0:
                gt_scores = gt_scores[hard_index].view(-1, 1).repeat(1, self.K)
                max_novel_scores = max_novel_scores[hard_index]
                assert(gt_scores.size() == max_novel_scores.size())
                assert(gt_scores.size(0) == hard_num)
                loss += self.loss_fn3(gt_scores.view(-1, 1), \
                        max_novel_scores.view(-1, 1), torch.ones(hard_num*self.K, 1).to(self.device)) * self.lw_mr

        pred = torch.argmax(logit, dim=1)

        acc = torch.sum(pred == y).item()
        return pred, acc / x.size(0), loss

    def after_task(self, task_idx, buffer, train_loader, test_loaders):
        if self.task_idx > 0:
            self.handle_ref_features.remove()
            self.handle_cur_features.remove()
            self.handle_old_scores_bs.remove()
            self.handle_new_scores_bs.remove()


    def inference(self, data):
        x, y = data['image'], data['label']
        x = x.to(self.device)
        y = y.to(self.device)
        
        logit = self.backbone(x)

        pred = torch.argmax(logit, dim=1)

        acc = torch.sum(pred == y).item()
        return pred, acc / x.size(0)


    def _init_optim(self, config, task_idx):
        if task_idx > 0:
            #fix the embedding of old classes
            ignored_params = list(map(id, self.backbone.fc.fc1.parameters()))
            base_params = filter(lambda p: id(p) not in ignored_params, \
                    self.backbone.parameters())
            tg_params =[{'params': base_params, 'lr': 0.1, 'weight_decay': 5e-4}, \
                        {'params': self.backbone.fc.fc1.parameters(), 'lr': 0, 'weight_decay': 0}]
        elif config['classifier']['name'] == 'LUCIR':
            tg_params = self.backbone.parameters()

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

### 数据划分相关参数
`data_root`: 数据集存放路径  <br>
`image_size`: 图片大小 <br>
`save_path`: 训练日志存放路径<br>
`init_cls_num`: 第一个任务类别数<br>
`inc_cls_num`: 其他任务类别数<br>
`task_num`: 任务数量<br>
```
data_root: /data/fanzhichen/continual/cifar100
image_size: 32
save_path: ./
init_cls_num: 50
inc_cls_num: 10
task_num: 6
```

### 训练优化器相关参数
关于训练模型所用的`optimizer`和`scheduler`。
`name`: 优化器的种类  <br>
`kwargs`: 按照`torch.optim`所用参数进行编写<br>
```
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
相关参数含义如优化器参数
```
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
`strategy`：选择`herding`更新策略，目前可支持`random`,`equal_random`,`reservoir`,`herding` <br>
```
buffer:
  name: LinearBuffer
  kwargs:
    buffer_size: 2000
    batch_size: 128
    strategy: herding     # random, equal_random, reservoir, herding
```


### 算法相关参数
`name`：此处标识所采用何种算法
`kwargs`: 算法z执行所需参数，会在`init_model`时传入。
```
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
