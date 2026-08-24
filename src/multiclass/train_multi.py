import sys
from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.multiclass.multi_model import MultiClassModel, loss_fn
from src.multi_preprocessing import train_loader, val_loader, CLASS_NAMES

torch.manual_seed(42)

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "model.pth"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

MAX_EPOCHS = 30
PATIENCE = 5  # stop after this many epochs with no val F1 improvement
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


use_amp = torch.cuda.is_available()
scaler = torch.amp.GradScaler(enabled=use_amp)  # prevents numerical errors

model = MultiClassModel(num_classes=len(CLASS_NAMES), freeze_backbone=False).to(device)

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
    epochs=MAX_EPOCHS,
    steps_per_epoch=len(train_loader)
)


def train_one_epoch() -> tuple:
    model.train()
    train_loss, train_acc = 0.0, 0.0
    for images, labels in train_loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = loss_fn(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # update the learning rate every batch
        scheduler.step()
        train_loss += loss.item()
        train_acc += MultiClassModel.multiClass_acc(logits.detach(), labels)

    return train_loss / len(train_loader), train_acc / len(train_loader)


@torch.inference_mode()
def evaluate(loader) -> dict:
    model.eval()
    total_loss = 0.0
    all_logits, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            total_loss += loss_fn(logits, labels).item()
        all_logits.append(logits)
        all_labels.append(labels)

    metrics = MultiClassModel.multiClass_metrics(torch.cat(all_logits), torch.cat(all_labels))
    metrics["Loss"] = total_loss / len(loader)
    return metrics


def train_loop() -> None:
    best_val_f1 = -1.0
    epochs_without_improvement = 0

    for epoch in range(MAX_EPOCHS):
        train_loss, train_acc = train_one_epoch()
        val_metrics = evaluate(val_loader)

        print(
            f"Epoch {epoch + 1}/{MAX_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_metrics['Loss']:.4f} | Val Acc: {val_metrics['Accuracy'] * 100:.2f}% | "
            f"Val F1: {val_metrics['F1']:.4f}"
        )

        if val_metrics["F1"] > best_val_f1:
            best_val_f1 = val_metrics["F1"]
            epochs_without_improvement = 0
            torch.save(model.state_dict(), MODEL_PATH)
            logging.info(f"Val F1 improved to {best_val_f1:.4f} -- saved model to {MODEL_PATH}")
        else:
            epochs_without_improvement += 1
            logging.info(f"No val F1 improvement for {epochs_without_improvement} epoch(s)")
            if epochs_without_improvement >= PATIENCE:
                logging.info(f"Early stopping at epoch {epoch + 1} (best val F1: {best_val_f1:.4f})")
                break

    logging.info(f"Training complete. Best model (val F1={best_val_f1:.4f}) saved to {MODEL_PATH}")


if __name__ == '__main__':
    logging.info(f"Using device: {device}")
    train_loop()
