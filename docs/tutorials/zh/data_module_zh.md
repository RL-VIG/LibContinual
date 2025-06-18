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

训练图像、测试图像需要分别放置在`train`和`test`文件夹下，其中同一类别所有图像放置在与类别同名文件夹中，例如`cat`、`dog`等。

## 配置数据集

当下载好或按照上述格式整理好数据集后，只需要在配置文件中修改`data_root`字段即可，注意`LibeContinual`会将数据集文件夹名当作数据集名称打印在log上。
