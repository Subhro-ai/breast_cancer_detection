import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

from src.dataset import MammogramDataset
from src.transform import get_transforms
from src.model import get_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 8
EPOCHS_PHASE1 = 7
EPOCHS_PHASE2 = 20
LR = 1e-4

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "SPLIT"

# ================= DATA =================
train_dataset = MammogramDataset(DATA_DIR / "train", transform=get_transforms(True))
val_dataset = MammogramDataset(DATA_DIR / "val", transform=get_transforms(False))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ================= MODEL =================
model = get_model().to(DEVICE)

criterion = nn.BCEWithLogitsLoss()

# ================= PHASE 1: Freeze Backbone =================
print("Phase 1: Training classifier only")

for param in model.parameters():
    param.requires_grad = False

for param in model.classifier.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

best_val_loss = float("inf")

def train_one_epoch():
    model.train()
    running_loss = 0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)


def validate():
    model.eval()
    running_loss = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

    return running_loss / len(val_loader)


# -------- Phase 1 Training --------
for epoch in range(EPOCHS_PHASE1):

    train_loss = train_one_epoch()
    val_loss = validate()

    scheduler.step(val_loss)

    print(f"[Phase1] Epoch {epoch+1}/{EPOCHS_PHASE1}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pth")
        print("Model saved!")

# ================= PHASE 2: Unfreeze Last Dense Block =================
print("\nPhase 2: Unfreezing last DenseBlock")

for name, param in model.named_parameters():
    if "denseblock4" in name:
        param.requires_grad = True

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)

# -------- Phase 2 Training --------
for epoch in range(EPOCHS_PHASE2):

    train_loss = train_one_epoch()
    val_loss = validate()

    scheduler.step(val_loss)

    print(f"[Phase2] Epoch {epoch+1}/{EPOCHS_PHASE2}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pth")
        print("Model saved!")

print("Training complete.")
