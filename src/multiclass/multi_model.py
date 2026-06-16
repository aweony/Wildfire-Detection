import torch
from torch import nn
from torchvision import models
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

class MultiClassModel(nn.Module):
    def __init__(self, num_classes: int, freeze_backbone=False):
        super().__init__()
        # loads a pre trained model
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Gets the number of input features going into ResNet's original final layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

    # evaluate the accuracy of the model
    @staticmethod
    def multiClass_acc(outputs, labels) -> float:
        preds = torch.argmax(outputs, dim=1)
        correct = torch.eq(preds, labels).sum().item()
        return (correct / len(labels)) * 100

    # evaluate the metrics of the model
    @staticmethod
    def multiClass_metrics(outputs, labels) -> dict:
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        labels_np = labels.cpu().numpy()
        return {
            "Accuracy": accuracy_score(labels_np, preds),
            "Precision": precision_score(labels_np, preds, average="weighted", zero_division=0),
            "Recall": recall_score(labels_np, preds, average="weighted", zero_division=0),
            "F1": f1_score(labels_np, preds, average="weighted", zero_division=0),
            "Confusion": confusion_matrix(labels_np, preds),
        }
