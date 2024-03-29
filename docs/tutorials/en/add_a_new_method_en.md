# Add a new method

Taking the [`LUCIR`](https://openaccess.thecvf.com/content_CVPR_2019/html/Hou_Learning_a_Unified_Classifier_Incrementally_via_Rebalancing_CVPR_2019_paper.html) method as an example, we will describe how to add a new method.

Before this, we need to introduce a parent class of all methods:`Finetune`.

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
The `Finetune` class includes several important interfaces that a method should have.
+ `__init__`: init func，set the initialize parameters required by the algorithm.
+ `observe`：used to be called in train phase, input a batch of training samples and return predictions, accuracy, and forward loss.
+ `inference`：used to be called in inference phase, input a batch of test samples and return the classification result and accuracy.
+ `forward`：override the forward function `forward` of `Module` in `pytorch`, return the ouput of `backbone`.
+ `before_task`：called before training starts for each task, used to adjust model structure, training parameters, etc., and requires user customization.
+ `after_task`：called after training starts for each task, used to adjust model structure, buffer, etc., and requires user customization.
+ `get_parameters`：called before training starts for each task, returns the training parameters for the current task.


## LUCIR

### Build model
First, create `LUCIR` model class, add file `lucir.py` under core/model/replay/: (this code have some differences with source code)
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
+ In `__init__`, initialize `K, lw_mr, ref_model` required by `LUCIR`.
+ In `before_task`, according to the requirements of `LUCIR`, we update the classifier before the task starts and set different loss functions based on `task_idx`.
+ In `observe`，we use the loss function defined in `before_task` to calculate the forward loss.
+ In `after_task`, according to the `LUCIR` algorithm, some `hook` operations need to be removed.
+ In `_init_optim`, we select a subset of parameters from the entire model for training.


The implementation of the above interfaces is the difference between the `LUCIR` algorithm and other algorithms. Other interfaces can be left unimplemented and handled by Finetune. <br>

Note that due to the distinct operations of continual learning algorithms for the first task and subsequent tasks, `task_idx` is passed in before_task to identify the current task number. <br>



## Add `lucir.yaml`

Please refer to [`config.md`](./config_file_en.md) for the meaning of each parameter
### Dataset

```yaml
data_root: /data/fanzhichen/continual/cifar100
image_size: 32
save_path: ./
init_cls_num: 50
inc_cls_num: 10
task_num: 6
```

### Optimizer

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

### Backbone
```yaml
backbone:
  name: resnet32
  kwargs:
    num_classes: 100
    args: 
      dataset: cifar100
      cosine_fc: True
```

### Buffer
`name`: `LinearBuffer` will merge the data with the current task data before the task starts.  <br>
`strategy`：Buffer update strategy, only support `herding, random, equal_random, reservoir, None` <br>
```yaml
buffer:
  name: LinearBuffer
  kwargs:
    buffer_size: 2000
    batch_size: 128
    strategy: herding     # random, equal_random, reservoir, herding
```


### Algorithm
`name`：which method. <br>
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
