"""
@inproceedings{DBLP:conf/iccv/ShiY23,
  title        = {Prototype Reminiscence and Augmented Asymmetric Knowledge Aggregation for Non-Exemplar Class-Incremental Learning},
  author       = {Shi, Wuxuan and Ye, Mang},
  booktitle    = {2023 IEEE/CVF International Conference on Computer Vision (ICCV)},
  pages        = {1772-1781},
  publisher    = {Computer Vision Foundation / {IEEE}},
  year         = {2023}
}

https://openaccess.thecvf.com/content/ICCV2023/papers/Shi_Prototype_Reminiscence_and_Augmented_Asymmetric_Knowledge_Aggregation_for_Non-Exemplar_Class-Incremental_ICCV_2023_paper.pdf

Adapted from https://github.com/ShiWuxuan/PRAKA
"""

from torch.nn import functional as F
import os
import numpy as np
import torch
import torch.nn as nn
import math
import copy
from core.model import Finetune

class joint_network(nn.Module):
    def __init__(self, numclass, feature_extractor):
        '''
        Code Reference:
        https://github.com/ShiWuxuan/PRAKA/blob/master/Cifar100/myNetwork.py
        '''
        super(joint_network, self).__init__()
        self.feature = feature_extractor
        self.fc = nn.Linear(512, numclass * 4, bias=True)
        self.classifier = nn.Linear(512, numclass, bias=True)

    def forward(self, input):
        '''
        Code Reference:
        https://github.com/ShiWuxuan/PRAKA/blob/master/Cifar100/myNetwork.py
        '''
        x = self.feature(input)
        x = self.classifier(x)
        return x

    def Incremental_learning(self, numclass):
        '''
        Update the fully connected (fc) layer and classifier layer to accommodate the new number of classes.

        This function modifies the output dimensions of the model's fully connected layer (`fc`)
        and the classifier layer based on the total number of classes after the current task.
        It ensures that the new layers retain the weights and biases from the previous configuration
        for the classes that were previously learned.

        Parameters:
        - numclass (int): The total number of classes after the current task, including both old and new classes.

        Notes:
        - The `fc` layer's output dimension is set to `numclass * 4`.
        - The classifier layer is adjusted to match the new total number of classes, while retaining the previously learned weights and biases.

        Code Reference:
        https://github.com/ShiWuxuan/PRAKA/blob/master/Cifar100/myNetwork.py
        '''
        weight = self.fc.weight.data
        bias = self.fc.bias.data
        in_feature = self.fc.in_features
        out_feature = self.fc.out_features

        self.fc = nn.Linear(in_feature, numclass * 4, bias=True)
        self.fc.weight.data[:out_feature] = weight[:out_feature]
        self.fc.bias.data[:out_feature] = bias[:out_feature]

        weight = self.classifier.weight.data
        bias = self.classifier.bias.data
        in_feature = self.classifier.in_features
        out_feature = self.classifier.out_features

        self.classifier = nn.Linear(in_feature, numclass, bias=True)
        self.classifier.weight.data[:out_feature] = weight[:out_feature]
        self.classifier.bias.data[:out_feature] = bias[:out_feature]

    def feature_extractor(self, inputs):
        '''
        Code Reference:
        https://github.com/ShiWuxuan/PRAKA/blob/master/Cifar100/myNetwork.py
        '''
        return self.feature(inputs)

class PRAKA(nn.Module):
    def __init__(self, backbone, feat_dim, num_class, **kwargs):
        #super().__init__(backbone, feat_dim, num_class, **kwargs)
        super().__init__()
        self.device = kwargs['device']
        self.kwargs = kwargs
        self.size = 32
        # Initialize the feature extractor with a custom ResNet18 structure.
        encoder = backbone
        self.model = joint_network(kwargs["init_cls_num"], encoder)
        self.radius = 0
        self.prototype = None
        self.numsamples = None
        self.numclass = kwargs["init_cls_num"]
        self.task_size = kwargs["inc_cls_num"]
        self.old_model = None
        # save the model and its corresponding task_id
        self.task_idx = 0

    def before_task(self, task_idx, buffer, train_loader, test_loaders):
        self.task_idx = task_idx
        if task_idx > 0:
            self.model.Incremental_learning(self.numclass)
        self.model.to(self.device)
        
    def observe(self, data):
        '''
            Processes a batch of training data to compute predictions, accuracy, and loss.

            Parameters:
            - data: Dictionary containing the batch of training samples
              - 'image': Tensor of input images
              - 'label': Tensor of ground truth labels

            Returns:
            - predictions: Tensor of predicted class labels for the input images
            - accuracy: Float value representing the accuracy of the model on the current batch
            - loss: Float value representing the computed loss for the batch

            Description:
            This function is called during the training phase. It performs the following steps:
            1. Extracts the images and labels from the provided data dictionary and transfers them to the device.
            2. Augments the images by rotating them by 0, 90, 180, and 270 degrees, and creates corresponding labels for these augmented images.
            3. Computes the loss using the augmented images and labels.
            4. Evaluates the model's performance on the current batch by calculating the accuracy and loss.
            5. Returns the predictions, accuracy, and loss for the batch.

            Example Usage:
            predictions, accuracy, loss = observe(data)
        '''
        images, labels = data['image'].to(self.device), data['label'].to(self.device)

        # Generate four times the number of images by rotating each image 0°, 90°, 180°, and 270°.
        images = torch.stack([torch.rot90(images, k, (2, 3)) for k in range(4)], 1)
        images = images.view(-1, 3, self.size, self.size)
        # Generate corresponding labels for the rotated images, each original label produces four new labels.
        joint_labels = torch.stack([labels * 4 + k for k in range(4)], 1).view(-1)
        if self.task_idx == 0:
            old_class = 0
        else:
            old_class = self.kwargs['init_cls_num'] + self.kwargs['inc_cls_num'] * (self.task_idx - 1)
        # Compute loss and predictions for a batch
        loss, single_preds = self._compute_loss(images, joint_labels, labels, old_class)

        preds = torch.argmax(single_preds, dim=-1)
        return preds, (preds == labels).sum().item() / len(labels), loss

    def inference(self, data):
        '''
            Performs inference on a batch of test samples and computes the classification results and accuracy.

            Parameters:
            - data: Dictionary containing the batch of test samples
              - 'image': Tensor of input images
              - 'label': Tensor of ground truth labels

            Returns:
            - predictions: Tensor of predicted class labels for the input images
            - accuracy: Float value representing the accuracy of the model on the current batch

            Example Usage:
            predictions, accuracy = inference(data)
        '''

        imgs, labels = data['image'].to(self.device), data['label'].to(self.device)

        preds = torch.argmax(self.model(imgs), dim=-1)

        return preds, (preds == labels).sum().item() / len(labels)

    def _compute_loss(self, imgs, joint_labels, labels, old_class=0):
        '''
            Computes the loss for a batch of images and labels.

            Parameters:
            - imgs: Tensor of input images
            - joint_labels: Tensor of labels for images augmented with rotations (0°, 90°, 180°, 270°)
            - labels: Tensor of ground truth labels for the images
            - old_class: Integer indicating the number of old classes (default is 0)

            Returns:
            - loss: Scalar tensor representing the total computed loss
            - preds: Tensor of predictions for the original (non-augmented) images

            Example Usage:
            loss, preds = self._compute_loss(imgs, joint_labels, labels, old_class)


            Code Reference:
            https://github.com/ShiWuxuan/PRAKA/blob/master/Cifar100/jointSSL.py
        '''
        # Feature extraction
        feature = self.model.feature(imgs)

        # Classification predictions
        joint_preds = self.model.fc(feature)
        single_preds = self.model.classifier(feature)[::4]
        joint_preds, joint_labels, single_preds, labels = joint_preds.to(self.device), joint_labels.to(self.device), single_preds.to(self.device), labels.to(self.device)
        joint_loss = nn.CrossEntropyLoss()(joint_preds/self.kwargs["temp"], joint_labels)
        single_loss = nn.CrossEntropyLoss()(single_preds/self.kwargs["temp"], labels)

        # Average loss for images generated by rotating 4 angles
        agg_preds = 0
        for i in range(4):
            agg_preds = agg_preds + joint_preds[i::4, i::4] / 4
        # Compute distillation loss between single predictions and aggregated predictions
        distillation_loss = F.kl_div(F.log_softmax(single_preds, 1),
                                    F.softmax(agg_preds.detach(), 1),
                                    reduction='batchmean')
        if old_class == 0:
            return joint_loss + single_loss + distillation_loss, single_preds
        else:
            feature_old = self.old_model.feature(imgs)

            loss_kd = torch.dist(feature, feature_old, 2)

            # Prototype augmentation
            proto_aug = []
            proto_aug_label = []
            old_class_list = list(self.prototype.keys())
            for _ in range(feature.shape[0] // 4):  # batch_size = feature.shape[0] // 4
                i = np.random.randint(0, feature.shape[0])
                np.random.shuffle(old_class_list)
                lam = np.random.beta(0.5, 0.5)
                if lam > 0.6:
                    lam = lam * 0.6

                if np.random.random() >= 0.5:
                    # Weighted combination of prototype (fixed image from old dataset) and current feature
                    temp = (1 + lam) * self.prototype[old_class_list[0]] - lam * feature.detach().cpu().numpy()[i]
                else:
                    temp = (1 - lam) * self.prototype[old_class_list[0]] + lam * feature.detach().cpu().numpy()[i]

                # Append the generated augmented features and corresponding labels to proto_aug and proto_aug_label
                proto_aug.append(temp)
                proto_aug_label.append(old_class_list[0])

            proto_aug = torch.from_numpy(np.float32(np.asarray(proto_aug))).float().to(self.device)
            proto_aug_label = torch.from_numpy(np.asarray(proto_aug_label)).to(self.device)
            aug_preds = self.model.classifier(proto_aug)
            joint_aug_preds = self.model.fc(proto_aug)
            agg_preds = joint_aug_preds[:, ::4]
            aug_distillation_loss = F.kl_div(F.log_softmax(aug_preds, 1),
                                            F.softmax(agg_preds.detach(), 1),
                                            reduction='batchmean')
            # Calculate the weighted sum of cross-entropy loss and distillation loss for augmented data
            loss_protoAug = nn.CrossEntropyLoss()(aug_preds/self.kwargs["temp"], proto_aug_label) + nn.CrossEntropyLoss()(joint_aug_preds/self.kwargs["temp"], proto_aug_label*4) + aug_distillation_loss
            return joint_loss + single_loss + distillation_loss + self.kwargs["protoAug_weight"]*loss_protoAug + self.kwargs["kd_weight"]*loss_kd, single_preds

    def after_task(self, task_idx, buffer, train_loader, test_loaders):
        '''
            Perform operations after completing the training for a specific task.
                    1. Save the prototypes of the current model.
                    2. Save the current model state to a file.
                    3. Load the saved model state as the old model for future reference.

            Parameters:
            - task_idx (int): The index of the current task.
            - buffer: Data buffer for storing samples (not used in this function).
            - train_loader (DataLoader): DataLoader for the training dataset of the current task.
            - test_loaders (list of DataLoader): List of DataLoaders for test datasets of different tasks.

            Example Usage:
            self.after_task(task_idx, buffer, train_loader, test_loaders)
        '''
        # Save the prototype
        self.protoSave(self.model, train_loader, self.task_idx)
        self.numclass += self.task_size

        self.old_model = copy.deepcopy(self.model)
        self.old_model.eval()

    def protoSave(self, model, loader, current_task):
        '''
            Save the class prototypes for the current task.

            This function extracts features from the data using the provided model and computes
            class prototypes based on these features. The prototypes are then saved to the class
            attributes. If it's the first task, the prototypes are initialized. For subsequent
            tasks, the prototypes are updated with new class information.

            Parameters:
            - model: The model used for feature extraction.
            - loader: DataLoader providing the dataset for the current task.
            - current_task (int): The index of the current task.

            Code Reference:
            https://github.com/ShiWuxuan/PRAKA/blob/master/Cifar100/jointSSL.py
        '''

        features = []
        labels = []
        model.eval()
        # Feature extraction
        with torch.no_grad():
            for i, batch in enumerate(loader):
                images, target = batch['image'], batch['label']
                feature = model.feature(images.to(self.device))
                if feature.shape[0] == loader.batch_size:
                    labels.append(target.numpy())
                    features.append(feature.cpu().numpy())

        labels_set = np.unique(labels)
        labels = np.array(labels)
        labels = np.reshape(labels, labels.shape[0] * labels.shape[1])
        features = np.array(features)
        features = np.reshape(features, (features.shape[0] * features.shape[1], features.shape[2]))

        # Compute class prototypes
        prototype = {}
        class_label = []
        numsamples = {}

        for item in labels_set:
            index = np.where(item == labels)[0]
            class_label.append(item)
            feature_classwise = features[index]
            prototype[item] = np.mean(feature_classwise, axis=0)
            # Record the number of samples for each class.
            numsamples[item] = feature_classwise.shape[0]
        if current_task == 0:
            self.prototype = prototype
            self.class_label = class_label
            self.numsamples = numsamples
        else:
            self.prototype.update(prototype)
            self.class_label = np.concatenate((class_label, self.class_label), axis=0)
            self.numsamples.update(numsamples)

    def get_parameters(self, config):
        return self.model.parameters()

