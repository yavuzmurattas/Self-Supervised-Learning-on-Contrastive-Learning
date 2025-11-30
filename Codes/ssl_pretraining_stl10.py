import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.datasets import STL10
from torchvision.models import resnet18

# ============================
# Configurations
# ============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

DATA_ROOT = "./data"
BATCH_SIZE_SSL = 256
EPOCHS_SSL = 50
IMAGE_SIZE = 96
TEMPERATURE = 0.5
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PROJECTION_DIM = 128
CHECKPOINT_PATH = r"Weights\resnet18_ssl_stl10.pth"

# ============================
# Evaluation Metric
# ============================
ssl_train_losses = []

# ============================
# Transform like SimCLR
# ============================

class SimCLRTransform:
    def __init__(self, image_size=96):
        color_jitter = transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.1
        )

        self.base_transform = transforms.Compose([
            # It is wanted to produce 2 different "strong" augments from the same image
            transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            # Classical ImageNet Normalizaiton for ResNet
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, x):
        # It is produced two different appearance from the same input x
        x_i = self.base_transform(x)
        x_j = self.base_transform(x)
        return x_i, x_j


# ============================
# Model: ResNet18 + Projection Head
# ============================

class ResNetSimCLR(nn.Module):
    def __init__(self, projection_dim=128):
        super().__init__()
        # This is the ResNet-18 model already embedded in torchvision
        self.encoder = resnet18(weights=None)
        feat_dim = self.encoder.fc.in_features  # generally 512

        # Projection head: h -> z (MLP of the SimCLR)
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, projection_dim),
        )

    def forward(self, x):
        # Forward Propagation of the ResNet-18 is written manually in order to not use the Fully Connected (FC) Layer
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)

        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        x = self.encoder.layer4(x)

        x = self.encoder.avgpool(x)
        h = torch.flatten(x, 1)  # (B, feat_dim) -> "representation"

        z = self.projector(h)
        # It is normalized to use cosine similarity in Contrastive loss.
        z = nn.functional.normalize(z, dim=1)

        return h, z


# ============================
# NT-Xent (InfoNCE) Loss
# ============================

def nt_xent_loss(z_i, z_j, temperature=0.5):
    # z_i and z_j: positive pair embeddings in (B, D) shape
    batch_size = z_i.size(0)
    z = torch.cat([z_i, z_j], dim=0)  # (2B, D)

    # Cosine similarity matrix: (2B, 2B)
    sim = torch.matmul(z, z.T)  # Since z is normalized, dot product = cosine
    sim = sim / temperature

    # Mask the Diagonal (self-sim) elements
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(mask, -9e15)

    # Determine the positive indexes
    # Positive partner for the first B: i + B
    # Negative partner for the last B: i - B
    pos_indices = torch.arange(2 * batch_size, device=z.device)
    pos_indices = (pos_indices + batch_size) % (2 * batch_size)

    # Positive similarities: sim[i, pos_i]
    pos_sim = sim[torch.arange(2 * batch_size, device=z.device), pos_indices]

    # -log( exp(pos) / sum(exp(all)) ) = -pos + logsumexp(all)
    loss = -pos_sim + torch.logsumexp(sim, dim=1)
    loss = loss.mean()

    return loss


# ============================
# DataLoader: STL-10 unlabeled
# ============================

def get_unlabeled_loader(batch_size=BATCH_SIZE_SSL):
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
        drop_last=True,
    )
    return loader


# ============================
# Evaluation Metric
# ============================
def save_train_loss():
    # ---- SSL: Train Loss vs Epoch ----
    epochs = range(1, len(ssl_train_losses) + 1)

    plt.figure()
    plt.plot(epochs, ssl_train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title("SSL Pretraining - Train Loss vs Epoch (ResNet-18, STL-10)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./Evaluation_Metrics/PreTrained_Evaluation_Metric/ssl_train_loss.png", dpi=150)
    plt.close()

# ============================
# Training
# ============================

def train_ssl():
    train_loader = get_unlabeled_loader()

    model = ResNetSimCLR(projection_dim=PROJECTION_DIM).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    model.train()
    global_step = 0

    for epoch in range(1, EPOCHS_SSL + 1):
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

        avg_loss = running_loss / len(train_loader)
        ssl_train_losses.append(avg_loss)
        print(f"[SSL] Epoch [{epoch}/{EPOCHS_SSL}] - Loss: {avg_loss:.4f}")

    # Weights in the model is saved after the tranining was finished.
    os.makedirs(os.path.dirname(CHECKPOINT_PATH) or ".", exist_ok=True)
    torch.save(model.encoder.state_dict(), CHECKPOINT_PATH)
    print(f"Self-supervised encoder weights recorded: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    train_ssl()
    save_train_loss()
