# This section introduces the flow control of the code.

The flow control process involves the following files:
- `run_trainer.py`: The outermost entry point of the program.
- `trainer.py`: The implementation file of the Trainer class, used to implement the training process of the model.
- `model.py`: The model file located in the `./core/model` folder, used to implement specific algorithm models.

## Entry Point

At the very beginning, the outermost logic execution of the code is in `run_trainer.py`. In this file, we initialize the trainer module and call its `train_loop` method to start the entire training process of the algorithm.

```python
# Initialization and calling of Trainer in run_trainer.py
trainer = Trainer(rank, config)
trainer.train_loop()
```

As follow, we will introduce [Initialization](#Initialization), [Loop control](#loop-control), [Task preprocessing](#task-preprocessing), [Model training](#model-training), [Post-task processing](#task-post-processing) and [Evaluation](#evaluation-process).



## Initialization

After the above initialization, we will get a `trainer` class. By calling the relevant methods of this class, we proceed with the subsequent model training.

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
During the initialization process of the trainer, we mainly initialize parameters such as the number of tasks, training rounds, training devices, log files, and result storage containers. For methods that require replay, we also initialize a buffer size. For methods that do not require replay, we initialize it to 0. In addition to initializing these necessary parameters, we also initialize the partitioning of the training and testing sets through the init_dataloader method. The meanings of the variables involved in this process are as follows:
- `config`: Save model related configuration parameters
- `Logger`: Storage of model logs
- `device`: specifies the device for model training
- ` init_data `: Set relevant data partitioning
- `model`: Save the model
- `buffer`: Possible memory replay
- `*meter`: Save relevant evaluation data

After the above initialization, we will obtain a `trainer` class, which can be used for subsequent model training by calling its related methods.

## Loop Control

After completing initialization, start the training process of the model by calling the `train_loop` method of `trainer`:
```python
class Trainer(object):
    def train_loop(self,):
        """
        The norm train loop:  before_task, train, test, after_task
        """
        pass
```
In this process, the first step is to call the model's [Task Preprocessing](#Task-Preprocessing), followed by [model training](#model-training). After the model training is completed, the model's [post task processing](#task-post-processing) is also called, and finally, [model evaluation](#evaluation-process) is performed. The following will further describe these processes.

## Task Preprocessing

In the task preprocessing process, the model will undergo some processing that may not be strongly related to model parameter optimization. For example, dynamically expanding related methods can initialize the network parameters that need to be expanded before the task. The specific implementation needs to be realized in the `before_task` method of each model file in the `model` module:

```python
# An example from the `./core/model/replay/finetune.py` file is shown below
class Finetune(nn.Module):
    def before_task(self, task_idx, buffer, train_loader, test_loaders):
        pass
```

## Model Training

Model training optimization is implemented through the `observe` method:

```python
class Trainer(object):
    def _train(self, epoch_idx, dataloader):
        ...
        output, acc, loss = self.model.observe(batch)
        ...
```
The method takes a batch of data and returns the logits, training accuracy, and training loss of the model's output. The model parameters are optimized by backpropagating through this loss. The specific implementation can refer to the content in `./core/model/replay/finetune.py`:
```python
# An example from the `./core/model/replay/finetune.py` file is shown below
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

## Task Post-processing

Similar to task preprocessing, task post-processing is used for some operations that may not be strongly related to model parameter optimization. For example, the method of re-issuing can update the replay memory in the post-task processing. The specific implementation needs to be realized in the `after_task` method of each model file in the `model` module:

```python
# An example from the `./core/model/replay/finetune.py` file is shown below
class Finetune(nn.Module):
    def after_task(self, task_idx, buffer, train_loader, test_loaders):
        pass
```

In addition, apart from some special operations, most operations that are not strongly related to model optimization can be processed either before or after the task, with the same effect.

## Evaluation Process

During the training process, the model's loss, training accuracy, and other metrics are saved in the `train_meter` for analysis:

```python
class Trainer(object):
    def train_loop(self,):
        ...
        train_meter = self._train(epoch_idx, dataloader)
        ...
```

In the evaluation phase of the model, the model is frozen and evaluated on the test set, and the results are saved in the `test_meter`. This is specifically implemented through the `_validate` method:

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
        return {"avg_acc": np.mean(per_task_acc), "per_task_acc": per_task_acc}
```