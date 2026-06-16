import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

MODEL_PATH = Path(__file__).parent.parent / "model.pth"

import torch
from multi_model import MultiClassModel, loss_fn
from preprocessing import train_loader, test_loader

torch.manual_seed(42)

EPOCHS = 25
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 4
model = MultiClassModel(num_classes=NUM_CLASSES, freeze_backbone=False).to(device)

# creating the optimizer
backbone_params = [p for name, p in model.backbone.named_parameters() if "fc" not in name]
head_params = list(model.backbone.fc.parameters())
optimizer = torch.optim.Adam([
    {"params": backbone_params, "lr": 1e-4},
    {"params": head_params, "lr": 1e-3},
], weight_decay=1e-4)

# changes the learning rate every batch for faster convergence
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=[1e-4, 1e-3],
    epochs=EPOCHS,
    steps_per_epoch=len(train_loader)
)

use_amp = torch.cuda.is_available()
scaler = torch.amp.GradScaler(enabled=use_amp)  # prevents numerical errors

def train_loop():
    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_acc = 0.0, 0.0
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images).squeeze()
                loss = loss_fn(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # update the learning rate every batch
            scheduler.step()
            # calculate the model loss and the model accuracy
            train_loss += loss.item()
            train_acc += MultiClassModel.multiClass_acc(logits.detach(), labels)

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

def test_loop():
    model.eval()
    test_loss, test_acc = 0.0, 0.0
    all_logits, all_labels = [], []
    with torch.inference_mode():
        for images, labels in test_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images).squeeze()
                test_loss += loss_fn(logits, labels).item()
            test_acc += MultiClassModel.multiClass_acc(logits, labels)
            all_logits.append(logits)
            all_labels.append(labels)

    test_loss /= len(test_loader)
    test_acc /= len(test_loader)

    # compute metrics across the full test set
    test_metrics = MultiClassModel.multiClass_metrics(
        torch.cat(all_logits), torch.cat(all_labels)
    )



