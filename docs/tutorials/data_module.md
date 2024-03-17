# 数据模块

## 本节相关代码：

```
core/data/augments.py
core/data/dataloader.py
core/data/dataset.py
```

## 数据集格式

在`LibContinual`中，所用的数据集有固定的格式。我们按照大多数持续学习设置下的数据集格式进行数据的读取，例如 [CIFAR-10](https://pytorch.org/vision/stable/datasets.html) 和 [CIFAR-100](https://pytorch.org/vision/stable/datasets.html) ，因此只需从网络上下载数据集并解压就可以使用。如果你想要使用一个新的数据集，并且该数据集的数据形式与以上数据集不同，那么你需要自己动手将其转换成相同的格式。

与 CIFAR-10 一样，数据集的格式应该和下面的示例一样：

```
dataset_folder/
├── train/
│   ├── class_1/
│      ├── image_1.png
│      ├── ...
│      └── image_5000.png
│   ├── ...
│   ├── class_10/
│      ├── image_1.png
│      ├── ...
│      └── image_5000.png
├── test/
│   ├── class_1/
│      ├── image_1.png
│      ├── ...
│      └── image_5000.png
│   ├── ...
│   ├── class_10/
│      ├── image_1.png
│      ├── ...
│      └── image_5000.png
```

所有的训练、验证以及测试图像都需要放置在`images`文件夹下，分别使用`train.csv`，`test.csv`和`val.csv`文件分割数据集。三个文件的格式都类似，需要以下面的格式进行数据的组织：

训练图像、测试图像需要分别放置在`train`和`test`文件夹下，其中同一类别所有图像放置在类别同名文件夹中，例如`cat`、`dog`等。

## 配置数据集

当下载好或按照上述格式整理好数据集后，只需要在配置文件中修改`data_root`字段即可，注意`LibeContinual`会将数据集文件夹名当作数据集名称打印在log上。

## 使用数据集和DataLoader

`DataLoader`用于处理持续学习数据集，为每个任务创建对象，并管理类别索引、数据增广和数据加载设置等参数。`DataLoader`会获取以下配置参数，并根据当前训练或测试模式选择对应的数据增广，用于后续数据集的划分与构建。

- `data_root`：数据集的根目录
- `task_num`：持续学习任务的总数
- `init_cls_num`：第一个任务的初始类别数量
- `inc_cls_num`：后续每个任务中类别数量

部分代码如下所示。首先，我们遍历由 `task_num` 指定的任务数量，根据 `init_cls_num` 和 `inc_cls_num` 计算每个任务所需数据的类别起始和结束索引。然后为每个任务创建各自的 `DataLoader` 对象，并将其添加到 `dataloaders` 列表中。

我们需要将每个任务各自的数据创建为`SingleDataset`对象，其使用相关参数进行实例化，包括指定的类别范围。

```python
def create_loaders(self):
	for i in range(self.task_num):
		start_idx = 0 if i == 0 else (self.init_cls_num + (i-1) * self.inc_cls_num)
		end_idx = start_idx + (self.init_cls_num if i ==0 else self.inc_cls_num)
		self.dataloaders.append(DataLoader(
			SingleDataseat(self.data_root, self.mode, self.cls_map, start_idx, end_idx, self.trfms),
				shuffle = True,
				batch_size = 32,
				drop_last = True
		))
```

