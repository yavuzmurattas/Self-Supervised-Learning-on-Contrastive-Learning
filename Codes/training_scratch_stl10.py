import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from torchvision import transforms
from torchvision.datasets import STL10
from torchvision.models import resnet18

# ============================
# Configurations
# ============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

DATA_ROOT = "./data"
BATCH_SIZE = 128
EPOCHS = 30
IMAGE_SIZE = 96
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_CLASSES = 10

SCRATCH_CHECKPOINT_PATH = r"Weights\resnet18_scratch_stl10.pth"

# ============================
# Evaluation Metrics
# ============================
train_losses = []
train_accs   = []
val_losses   = []
val_accs     = []
val_f1s      = []


# ============================
# Transforms
# ============================

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ============================
# Dataset and DataLoader
# ============================

def get_train_val_test_loaders(batch_size=BATCH_SIZE, val_ratio=0.2):
    # Labeled train split
    full_train_ds = STL10(
        root=DATA_ROOT,
        split="train",
        download=True,
        transform=train_transform,
    )

    # Test split
    test_ds = STL10(
        root=DATA_ROOT,
        split="test",
        download=True,
        transform=test_transform,
    )

    # Train/val split (For example 4000 train, 1000 val)
    num_train = len(full_train_ds)
    num_val = int(num_train * val_ratio)
    num_train = num_train - num_val
    train_ds, val_ds = random_split(full_train_ds, [num_train, num_val])

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader


# ============================
# Model: ResNet-18 (Scratch)
# ============================

def get_scratch_model():
    # weights=None -> completely random initialization
    model = resnet18(weights=None)

    # Change the last layer to 10 classes for STL-10
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, NUM_CLASSES)

    return model.to(device)


# ============================
# Accuracy and Macro F1
# ============================

def compute_accuracy(preds, labels):
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return correct / total


def macro_f1_score(y_true, y_pred, num_classes):
    # y_true and y_pred: 1D tensor (N,)
    f1s = []
    for c in range(num_classes):
        tp = ((y_pred == c) & (y_true == c)).sum().item()
        fp = ((y_pred == c) & (y_true != c)).sum().item()
        fn = ((y_pred != c) & (y_true == c)).sum().item()

        if tp == 0 and fp == 0 and fn == 0:
            f1s.append(0.0)
            continue

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        f1s.append(f1)

    if len(f1s) == 0:
        return 0.0
    return sum(f1s) / len(f1s)


# ============================
# Train & Eval Functions
# ============================

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size

        preds = outputs.argmax(dim=1)
        running_acc += (preds == labels).sum().item()
        total_samples += batch_size

    epoch_loss = running_loss / total_samples
    epoch_acc = running_acc / total_samples
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    total_samples = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size

            preds = outputs.argmax(dim=1)
            running_acc += (preds == labels).sum().item()
            total_samples += batch_size

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    epoch_loss = running_loss / total_samples
    epoch_acc = running_acc / total_samples

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    macro_f1 = macro_f1_score(all_labels, all_preds, NUM_CLASSES)

    return epoch_loss, epoch_acc, macro_f1


# ============================
# Evaluation Metric
# ============================
def save_train_loss():
    epochs = range(1, len(train_losses) + 1)

    # 1) Train Loss
    plt.figure()
    plt.plot(epochs, train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title("Scratch - Train Loss vs Epoch (ResNet-18, STL-10)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./Evaluation_Metrics/Train_Scratch_Metrics/scratch_train_loss.png", dpi=150)
    plt.close()

    # 2) Train Accuracy
    plt.figure()
    plt.plot(epochs, train_accs)
    plt.xlabel("Epoch")
    plt.ylabel("Train Accuracy")
    plt.title("Scratch - Train Accuracy vs Epoch (ResNet-18, STL-10)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./Evaluation_Metrics/Train_Scratch_Metrics/scratch_train_acc.png", dpi=150)
    plt.close()

    # 3) Validation Loss
    plt.figure()
    plt.plot(epochs, val_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Val Loss")
    plt.title("Scratch - Validation Loss vs Epoch (ResNet-18, STL-10)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./Evaluation_Metrics/Train_Scratch_Metrics/scratch_val_loss.png", dpi=150)
    plt.close()

    # 4) Validation Accuracy
    plt.figure()
    plt.plot(epochs, val_accs)
    plt.xlabel("Epoch")
    plt.ylabel("Val Accuracy")
    plt.title("Scratch - Validation Accuracy vs Epoch (ResNet-18, STL-10)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./Evaluation_Metrics/Train_Scratch_Metrics/scratch_val_acc.png", dpi=150)
    plt.close()

    # 5) Validation Macro-F1
    plt.figure()
    plt.plot(epochs, val_f1s)
    plt.xlabel("Epoch")
    plt.ylabel("Val Macro-F1")
    plt.title("Scratch - Validation Macro-F1 vs Epoch (ResNet-18, STL-10)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./Evaluation_Metrics/Train_Scratch_Metrics/scratch_val_macro_f1.png", dpi=150)
    plt.close()


# ============================
# Training (from Scratch)
# ============================

def train_scratch():
    train_loader, val_loader, test_loader = get_train_val_test_loaders()

    model = get_scratch_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)

        print(
            f"[SCRATCH] Epoch [{epoch}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Macro-F1: {val_f1:.4f}"
        )

        # Track best validation accuracy, save model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(SCRATCH_CHECKPOINT_PATH) or ".", exist_ok=True)
            torch.save(model.state_dict(), SCRATCH_CHECKPOINT_PATH)
            print(f"  -> New best scratch model saved (Val Acc = {best_val_acc:.4f})")

    # Evaluate the best model on the test set
    print("The best scratch model is evaluated on the test set...")
    best_model = get_scratch_model()
    best_model.load_state_dict(torch.load(SCRATCH_CHECKPOINT_PATH, map_location=device))
    best_model.to(device)

    test_loss, test_acc, test_f1 = evaluate(best_model, test_loader, criterion)
    print(f"[SCRATCH TEST] Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, Macro-F1: {test_f1:.4f}")


if __name__ == "__main__":
    train_scratch()
    save_train_loss()
