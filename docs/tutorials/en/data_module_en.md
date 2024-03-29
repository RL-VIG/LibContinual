# Data Module

## Related codes：

```
core/data/augments.py
core/data/dataloader.py
core/data/dataset.py
```

## Dataset file format

In `LibContinual`, the dataset used has a fixed format. We read the data according to the dataset format set by most continual learning settings, such as [CIFAR-10](https://pytorch.org/vision/stable/datasets.html) and [CIFAR-100](https://pytorch.org/vision/stable/datasets.html). So we only need to download the dataset from the network and decompress it to use. If you want to use a new dataset and its data format is different from the above datasets, you need to convert it to the same format yourself.

Like CIFAR-10, the file format of the dataset should be the same as the following example:

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

The training images and test images need to be placed in the `train` and `test` folders respectively, where all images of the same category are placed in folde with the same name as the category, such as `cat` , `dog`, etc.

## Configure Datasets

After downloading or organizing the dataset according to the above file format, simply modify the `data_root` field in the configuration file. Note that `LibeContinual` will print the dataset folder name as the dataset name on the log.
