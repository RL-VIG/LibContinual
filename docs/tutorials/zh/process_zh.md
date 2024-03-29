# 本节介绍代码的流程控制

流程控制过程涉及以下几个文件：
- `run_trainer.py`: 程序最外层的入口
- `trainer.py`: `Trainer`类别的实现文件，用来实现模型的训练流程
- `model.py`：位于`./core/model`文件夹下的模型文件，用于实现具体算法模型

## 入口
首先，代码执行逻辑的最外层是`run_trainer.py`，在这个文件中，我们通过初始化`trainer`模块后，调用它的`train_loop`方法开启真个算法的训练流程。
```python
# run_trainer.py中初始化和调用Trainer
trainer = Trainer(rank, config)
trainer.train_loop()
```
以下我们将从[初始化](#初始化)、[循环控制](#循环控制)、[任务前处理](#任务前处理)、[模型训练](#模型训练)、[任务后处理](#任务后处理)、[评估流程](#评估流程)几个方面展开说明。

## 初始化
首先，需要对训练器进行初始化，初始化的实现代码位于`trainer.py`文件中：
```python
class Trainer(object):
    """
    The Trainer.
    Build a trainer from config dict, set up optimizer, model, etc.
    """
    def __init__(self, rank, config):
        # initialize the Trainer
        pass
```
在训练器初始化的过程中，我们主要是初始化任务数量、训练轮次、训练设备、日志文件、结果存储容器等参数，需要重放的方法还要初始化一个buffer大小，对于不需要重放的方法就初始化为0。除了初始化这些必备的参数之外，还通过_init_dataloader方法初始化训练集和测试集的划分。这一过程中涉及到的变量含义如下：
- `config`: 保存模型相关的配置参数
- `logger`: 模型的日志存储
- `device`: 指定模型训练的设备
- `_init_data`: 设置相关的数据划分
- `model`: 保存模型
- `buffer`: 可能存在重放内存
- `*meter`: 保存相关的评估数据

经过以上的初始化，会得到一个`trainer`类，通过调用这个类的相关方法进行后面的模型训练。

## 循环控制
在完成初始化之后，通过调用`trainer`的`train_loop`方法开始模型的训练流程:
```python
class Trainer(object):
    def train_loop(self,):
        """
        The norm train loop:  before_task, train, test, after_task
        """
        pass
```
在这个过程中，首先对调用模型的[任务前处理](#任务前处理)，而后进行[模型训练](#模型训练)，在模型训练结束后还要调用模型的[任务后处理](#任务后处理)，最后进行[模型的评估](#评估流程)。下面将对这些过程进行进一步描述。

## 任务前处理
在任务前处理过程中，模型将进行一个和模型参数优化可能并没有强相关的一些处理。比如，动态拓展相关的方法，可以在任务前初始化需要拓展的网络参数。具体的实现，需要在model模块中，每个模型文件各自的before_task方法下实现:
```python
# 以./core/model/replay/finetune.py文件为例展示
class Finetune(nn.Module):
    def before_task(self, task_idx, buffer, train_loader, test_loaders):
        pass
```

## 模型训练
模型训练优化通过observe方法实现:
```python
class Trainer(object):
    def _train(self, epoch_idx, dataloader):
        ...
        output, acc, loss = self.model.observe(batch)
        ...
```
该方法输入一个batch的数据会返回模型输出的logits、训练精度、训练损失,通过对这个损失进行反向回传来优化模型参数。方法的具体实现可以参考`./core/model/replay/finetune.py`中的内容：
```python
# 以./core/model/replay/finetune.py文件为例展示
class Finetune(nn.Module):
    def observe(self, data):
        x, y = data['image'], data['label']
        x = x.to(self.device)
        y = y.to(self.device)
        logit = self.classifier(self.backbone(x)['features'])    
        loss = self.loss_fn(logit, y)
        pred = torch.argmax(logit, dim=1)
        acc = torch.sum(pred == y).item()
        return pred, acc / x.size(0), loss
```
## 任务后处理
和任务前处理相似，任务后处理用来进行一些和模型参数优化可能并不强相关的一些操作。比如，重放的方法可以在任务后处理中更新重放的内存。具体实现在每个模型文件的after_task方法中实现。具体的实现，需要在model模块中，每个模型文件各自的after_task方法下实现:
```python
# 以./core/model/replay/finetune.py文件为例展示
class Finetune(nn.Module):
    def after_task(self, task_idx, buffer, train_loader, test_loaders):
        pass
```
此外，除了一些较为特别的操作，大部分与模型优化不强相关的操作既可以放在任务前也可以放在任务后进行处理，效果是一样的。

## 评估流程
在训练过程中，模型的损失、训练精度等指标会被保存到`train_meter`中，用于分析：
```python
class Trainer(object):
    def train_loop(self,):
        ...
        train_meter = self._train(epoch_idx, dataloader)
        ...
```
在模型的评估阶段，将模型冻结后在测试集上评估，并将结果保存到`test_meter`中，具体通过`_validate`方法实现：
```python
class Trainer(object):
    def _validate(self, task_idx):
        dataloaders = self.test_loader.get_loader(task_idx)
        self.model.eval()
        meter = self.test_meter
        per_task_acc = []
        with torch.no_grad():
            for t, dataloader in enumerate(dataloaders):
                meter[t].reset()
                for batch_idx, batch in enumerate(dataloader):
                    output, acc = self.model.inference(batch)
                    meter[t].update("acc1", acc)
                per_task_acc.append(round(meter[t].avg("acc1"), 2))
        return {"avg_acc" : np.mean(per_task_acc), "per_task_acc" : per_task_acc}
```