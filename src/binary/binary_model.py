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

loss_fn = nn.BCEWithLogitsLoss()

class BinaryModel(nn.Module):
    def __init__(self, freeze_backbone=True):
        super().__init__()
        # loads a pre trained model
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        # Gets the number of input features going into ResNet’s original final layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.backbone(x)
    
    @staticmethod
    def binary_acc(outputs, labels):
        preds = (outputs.sigmoid().squeeze() > 0.5).float()
        correct = torch.eq(preds, labels).sum().item()
        return (correct / len(labels)) * 100

    # evaulate the metrics of the model
    @staticmethod
    def binary_metrics(outputs, labels) -> dict:
        preds = (outputs.sigmoid().squeeze() > 0.5).float()
        metrics = {}
        preds = (outputs.sigmoid().squeeze() > 0.5).float()
        metrics["Accuracy"] = accuracy_score(labels, preds)
        metrics["Precision"] = precision_score(labels, preds)
        metrics ["Recall"] = recall_score(labels, preds)
        metrics["f1"] = f1_score(labels, preds)
        metrics["confusion"] = confusion_matrix(labels, preds)
        return metrics


