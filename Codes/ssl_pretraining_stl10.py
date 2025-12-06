import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.datasets import STL10
from torchvision.models import resnet50

# ============================
# Configurations
# ============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

DATA_ROOT = "./data"

# SimCLR typically uses very large batch sizes (e.g., 4096).
# Here we use 256 for practicality.
BATCH_SIZE_SSL = 256

# Number of epochs for self-supervised pretraining.
# Original SimCLR often uses 100–800 epochs, but we keep it moderate here.
EPOCHS_SSL = 50

IMAGE_SIZE = 96          # STL-10 default is 96x96 (SimCLR uses 224x224 on ImageNet)
TEMPERATURE = 0.5

# SimCLR uses a base LR of 0.3, scaled linearly with batch size (B/256).
BASE_LR = 0.3
LEARNING_RATE = BASE_LR * BATCH_SIZE_SSL / 256.0

WEIGHT_DECAY = 1e-4
PROJECTION_DIM = 128

CHECKPOINT_PATH = r"Weights\resnet50_ssl_stl10.pth"

# ============================
# Tracking (SSL training loss per epoch)
# ============================
ssl_train_losses = []


# ============================
# SimCLR Data Augmentation
# ============================

class SimCLRTransform:
    """
    Apply two independent strong augmentations to the same input image,
    following the SimCLR augmentations:
      - RandomResizedCrop
      - RandomHorizontalFlip
      - ColorJitter
      - RandomGrayscale
      - GaussianBlur
      - Normalization
    """
    def __init__(self, image_size=96):
        # Color jitter parameters similar to SimCLR (brightness, contrast,
        # saturation = 0.8, hue = 0.2)
        color_jitter = transforms.ColorJitter(
            brightness=0.8,
            contrast=0.8,
            saturation=0.8,
            hue=0.2
        )

        self.base_transform = transforms.Compose([
            # Random resized crop with small minimum scale (0.08) as in SimCLR
            transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            # Gaussian blur is a key augmentation in SimCLR
            transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            # Standard ImageNet normalization for ResNet
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, x):
        """
        Return two differently augmented views from the same input image x.
        """
        x_i = self.base_transform(x)
        x_j = self.base_transform(x)
        return x_i, x_j


# ============================
# Model: ResNet-50 + SimCLR Projection Head
# ============================

class ResNetSimCLR(nn.Module):
    """
    ResNet-50 backbone + 2-layer MLP projection head, as in the original SimCLR.

    The backbone outputs a representation h (feature vector),
    and the MLP maps it to a projection z used in the contrastive loss.
    """
    def __init__(self, projection_dim=128):
        super().__init__()

        # ResNet-50 backbone (no pretrained weights, we will pretrain it with SimCLR)
        self.encoder = resnet50(weights=None)
        feat_dim = self.encoder.fc.in_features  # 2048 for ResNet-50

        # Replace the original FC layer by identity so encoder(x) returns feature representation
        self.encoder.fc = nn.Identity()

        # Projection head: MLP h -> z (2-layer MLP, hidden dim 2048 as in SimCLR)
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, projection_dim),
        )

    def forward(self, x):
        """
        Forward pass:
          - encoder(x) -> h (representation)
          - projector(h) -> z (projection)
          - normalize z for cosine similarity in the contrastive loss.
        """
        h = self.encoder(x)          # (B, feat_dim)
        z = self.projector(h)        # (B, projection_dim)
        z = F.normalize(z, dim=1)    # L2 normalization for cosine similarity
        return h, z


# ============================
# NT-Xent (InfoNCE) Loss
# ============================

def nt_xent_loss(z_i, z_j, temperature=0.5):
    """
    Normalized Temperature-scaled Cross Entropy Loss (NT-Xent) used in SimCLR.

    z_i and z_j are embeddings of positive pairs with shape (B, D).
    The loss is computed across the 2B concatenated embeddings.
    """
    batch_size = z_i.size(0)

    # Concatenate along batch dimension: (2B, D)
    z = torch.cat([z_i, z_j], dim=0)
    z = F.normalize(z, dim=1)  # extra safety normalization

    # Cosine similarity matrix (2B x 2B)
    sim = torch.matmul(z, z.T) / temperature  # logits

    # Mask out self-similarities on the diagonal
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=sim.device)
    sim = sim.masked_fill(mask, -9e15)  # large negative number ~ -inf

    # For each index i in [0, 2B), its positive pair is (i + B) mod (2B)
    targets = (torch.arange(2 * batch_size, device=sim.device) + batch_size) % (2 * batch_size)

    # Cross-entropy between similarity logits and targets (positive indices)
    loss = F.cross_entropy(sim, targets)
    return loss


# ============================
# DataLoader: STL-10 unlabeled split
# ============================

def get_unlabeled_loader(batch_size=BATCH_SIZE_SSL):
    """
    Return dataloader over STL-10 'unlabeled' split with SimCLR augmentations.
    """
    transform = SimCLRTransform(image_size=IMAGE_SIZE)
    unlabeled_ds = STL10(
        root=DATA_ROOT,
        split="unlabeled",
        download=True,
        transform=transform,
    )
    loader = DataLoader(
        unlabeled_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,  # important for SimCLR to keep batch sizes consistent
    )
    return loader


# ============================
# Plotting SSL Training Loss
# ============================

def save_train_loss():
    """
    Save SSL training loss curve (loss vs epoch) for visualization.
    """
    os.makedirs("./Evaluation_Metrics/PreTrained_Evaluation_Metric", exist_ok=True)
    epochs = range(1, len(ssl_train_losses) + 1)

    plt.figure()
    plt.plot(epochs, ssl_train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title("SSL Pretraining - Train Loss vs Epoch (ResNet-50, STL-10)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./Evaluation_Metrics/PreTrained_Evaluation_Metric/ssl_train_loss.png", dpi=150)
    plt.close()


# ============================
# Self-Supervised Training (SimCLR Pretraining)
# ============================

def train_ssl():
    """
    Perform SimCLR self-supervised pretraining on STL-10 unlabeled data.

    We train a ResNet-50 encoder with NT-Xent loss on pairs of augmented views
    from the same images. After training, only the encoder backbone weights
    are saved (for future fine-tuning / linear evaluation).
    """
    train_loader = get_unlabeled_loader()

    model = ResNetSimCLR(projection_dim=PROJECTION_DIM).to(device)


    # SimCLR uses LARS in the original implementation; here we approximate with SGD
    # and cosine learning rate schedule for simplicity.
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=0.9,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS_SSL,
        eta_min=0.0,
    )

    global_step = 0

    for epoch in range(1, EPOCHS_SSL + 1):
        model.train()
        running_loss = 0.0

        for (x_i, x_j), _ in train_loader:
            x_i = x_i.to(device)
            x_j = x_j.to(device)

            optimizer.zero_grad()

            _, z_i = model(x_i)
            _, z_j = model(x_j)

            loss = nt_xent_loss(z_i, z_j, temperature=TEMPERATURE)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            global_step += 1

        # Cosine LR update per epoch
        scheduler.step()

        avg_loss = running_loss / len(train_loader)
        ssl_train_losses.append(avg_loss)
        current_lr = scheduler.get_last_lr()[0]
        print(f"[SSL] Epoch [{epoch}/{EPOCHS_SSL}] - Loss: {avg_loss:.4f} - LR: {current_lr:.5f}")

    # After training, save only the encoder weights (ResNet-50 backbone).
    os.makedirs(os.path.dirname(CHECKPOINT_PATH) or ".", exist_ok=True)
    torch.save(model.encoder.state_dict(), CHECKPOINT_PATH)
    print(f"Self-supervised ResNet-50 encoder weights saved at: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    train_ssl()
    save_train_loss()
