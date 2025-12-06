import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from torchvision import transforms
from torchvision.datasets import STL10
from torchvision.models import resnet50

# ============================
# Configurations
# ============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

DATA_ROOT = "./data"
BATCH_SIZE_FT = 128
EPOCHS_FT = 30
IMAGE_SIZE = 96
LEARNING_RATE_FT = 1e-3
WEIGHT_DECAY_FT = 1e-4
NUM_CLASSES = 10

# SSL checkpoint path (encoder weights from SimCLR pretraining with ResNet-50)
SSL_CHECKPOINT_PATH = r"Weights\resnet50_ssl_stl10.pth"

# Path to store the best supervised model (linear eval / fine-tuned classifier)
FINETUNED_CHECKPOINT_PATH = r"Weights\resnet50_ssl_finetuned_stl10.pth"

# ============================
# Containers for Evaluation Metrics
# ============================
train_losses = []
train_accs   = []
val_losses   = []
val_accs     = []
val_f1s      = []   # Macro-F1


# ============================
# Transforms for Supervised Training
# ============================

# Data augmentation for supervised training (fine-tuning / linear eval)
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Deterministic transform for validation and test
test_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ============================
# Dataset and DataLoaders
# ============================

def get_train_val_test_loaders(batch_size=BATCH_SIZE_FT, val_ratio=0.2):
    """
    Create train / validation / test dataloaders for STL-10.

    The labeled 'train' split (5000 images) is further split into:
      - train subset
      - validation subset (val_ratio portion).
    """
    full_train_ds = STL10(
        root=DATA_ROOT,
        split="train",
        download=True,
        transform=train_transform,
    )
    test_ds = STL10(
        root=DATA_ROOT,
        split="test",
        download=True,
        transform=test_transform,
    )

    # e.g. ~4000 train, ~1000 val for STL-10 if val_ratio=0.2
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
# Model: ResNet-50 classifier for SimCLR linear evaluation
# ============================

def get_model(
    use_ssl_init: bool = True,
    ssl_ckpt_path: str = SSL_CHECKPOINT_PATH,
    freeze_backbone: bool = True,
):
    """
    Build a ResNet-50 model for STL-10 classification.

    - If use_ssl_init=True, load encoder weights from SimCLR pretraining
      (stored in ssl_ckpt_path).
    - Replace the final fully connected layer with a new Linear(NUM_CLASSES).
    - If freeze_backbone=True:
        => this corresponds to the original SimCLR "linear evaluation" protocol:
           all encoder layers are frozen, only the classifier (fc) is trained.
      If freeze_backbone=False:
        => this becomes a full fine-tuning setup (SimCLR initialization).
    """
    model = resnet50(weights=None)

    # Optionally initialize from SSL-pretrained encoder weights
    if use_ssl_init and os.path.exists(ssl_ckpt_path):
        print(f"Loading SSL encoder weights from: {ssl_ckpt_path}")
        state_dict = torch.load(ssl_ckpt_path, map_location=device)
        # These weights come from the SimCLR encoder (backbone only, fc is Identity).
        # strict=False to ignore any mismatch in the final classification layer.
        model.load_state_dict(state_dict, strict=False)
    else:
        print("SSL weights not found or not used. Continuing with random initialization.")

    # Replace the last fully connected layer to match the number of STL-10 classes
    in_features = model.fc.in_features  # 2048 for ResNet-50
    model.fc = nn.Linear(in_features, NUM_CLASSES)

    # Original SimCLR linear evaluation protocol freezes the backbone:
    # only the final linear layer is trained.
    if freeze_backbone:
        for name, param in model.named_parameters():
            # Train only parameters of 'fc' (the classifier head)
            if not name.startswith("fc."):
                param.requires_grad = False

    return model.to(device)


# ============================
# Accuracy and Macro-F1
# ============================

def compute_accuracy(preds, labels):
    """
    Compute classification accuracy.

    preds: tensor of predicted class indices, shape (N,)
    labels: tensor of ground-truth class indices, shape (N,)
    """
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return correct / total


def macro_f1_score(y_true, y_pred, num_classes):
    """
    Compute macro-averaged F1 score across all classes.

    y_true, y_pred: 1D tensors of size N containing class indices.
    num_classes: total number of classes.
    """
    f1s = []
    for c in range(num_classes):
        tp = ((y_pred == c) & (y_true == c)).sum().item()
        fp = ((y_pred == c) & (y_true != c)).sum().item()
        fn = ((y_pred != c) & (y_true == c)).sum().item()

        # If a class is completely absent (no positive or negative examples),
        # we can treat its F1 as zero.
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
# Train & Evaluation Functions
# ============================

def train_one_epoch(model, loader, criterion, optimizer):
    """
    Single-epoch supervised training loop.
    Returns average loss and accuracy over the epoch.
    """
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
    """
    Evaluation loop for validation or test sets.
    Returns average loss, accuracy and macro-F1 score.
    """
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
# Plotting Supervised Training Curves
# ============================

def save_train_loss():
    """
    Save fine-tuning / linear-eval curves:

    - Train loss
    - Train accuracy
    - Validation loss
    - Validation accuracy
    - Validation macro-F1
    """
    os.makedirs("./Evaluation_Metrics/FineTuning_Metrics", exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    # 1) Train Loss
    plt.figure()
    plt.plot(epochs, train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title("Supervised Training - Train Loss vs Epoch (ResNet-50, STL-10)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./Evaluation_Metrics/FineTuning_Metrics/ft_train_loss.png", dpi=150)
    plt.close()

    # 2) Train Accuracy
    plt.figure()
    plt.plot(epochs, train_accs)
    plt.xlabel("Epoch")
    plt.ylabel("Train Accuracy")
    plt.title("Supervised Training - Train Accuracy vs Epoch (ResNet-50, STL-10)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./Evaluation_Metrics/FineTuning_Metrics/ft_train_acc.png", dpi=150)
    plt.close()

    # 3) Validation Loss
    plt.figure()
    plt.plot(epochs, val_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Val Loss")
    plt.title("Supervised Training - Validation Loss vs Epoch (ResNet-50, STL-10)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./Evaluation_Metrics/FineTuning_Metrics/ft_val_loss.png", dpi=150)
    plt.close()

    # 4) Validation Accuracy
    plt.figure()
    plt.plot(epochs, val_accs)
    plt.xlabel("Epoch")
    plt.ylabel("Val Accuracy")
    plt.title("Supervised Training - Validation Accuracy vs Epoch (ResNet-50, STL-10)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./Evaluation_Metrics/FineTuning_Metrics/ft_val_acc.png", dpi=150)
    plt.close()

    # 5) Validation Macro-F1
    plt.figure()
    plt.plot(epochs, val_f1s)
    plt.xlabel("Epoch")
    plt.ylabel("Val Macro-F1")
    plt.title("Supervised Training - Validation Macro-F1 vs Epoch (ResNet-50, STL-10)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./Evaluation_Metrics/FineTuning_Metrics/ft_val_macro_f1.png", dpi=150)
    plt.close()


# ============================
# Fine-tuning / Linear Evaluation
# ============================

def finetune(use_ssl_init: bool = True, freeze_backbone: bool = True):
    """
    Supervised training after SimCLR:

    - If use_ssl_init=True:
        Load SimCLR-pretrained encoder weights as initialization.
    - If freeze_backbone=True (default):
        This is the original SimCLR "linear evaluation" protocol:
          * encoder is frozen
          * only the last linear layer (fc) is trained.
    - If freeze_backbone=False:
        This becomes full fine-tuning of the entire network.
    """
    train_loader, val_loader, test_loader = get_train_val_test_loaders()

    model = get_model(
        use_ssl_init=use_ssl_init,
        ssl_ckpt_path=SSL_CHECKPOINT_PATH,
        freeze_backbone=freeze_backbone,
    )

    criterion = nn.CrossEntropyLoss()

    # Optimize only parameters that require gradients.
    # For linear evaluation, this will be only model.fc parameters.
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE_FT,
        weight_decay=WEIGHT_DECAY_FT,
    )

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS_FT + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)

        print(
            f"[FT] Epoch [{epoch}/{EPOCHS_FT}] "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Macro-F1: {val_f1:.4f}"
        )

        # Track the best validation accuracy and save the model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(FINETUNED_CHECKPOINT_PATH) or ".", exist_ok=True)
            torch.save(model.state_dict(), FINETUNED_CHECKPOINT_PATH)
            print(f"  -> New best model recorded (Val Acc = {best_val_acc:.4f})")

    # Load the best model and evaluate it on the test set
    print("Evaluating the best model on the test set...")
    best_model = get_model(
        use_ssl_init=False,   # we will load weights from the checkpoint directly
        ssl_ckpt_path=SSL_CHECKPOINT_PATH,
        freeze_backbone=freeze_backbone,
    )
    best_model.load_state_dict(torch.load(FINETUNED_CHECKPOINT_PATH, map_location=device))
    best_model.to(device)

    test_loss, test_acc, test_f1 = evaluate(best_model, test_loader, criterion)
    print(f"[TEST] Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, Macro-F1: {test_f1:.4f}")


if __name__ == "__main__":
    # use_ssl_init=True  -> use SimCLR-pretrained encoder as initialization
    # freeze_backbone=True -> original SimCLR "linear evaluation" setup
    finetune(use_ssl_init=True, freeze_backbone=True)
    save_train_loss()
