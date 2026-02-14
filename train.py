import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

from src.dataset import MammogramDataset
from src.transform import get_transforms
from src.model import get_model


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 8
EPOCHS = 15
LR = 1e-4

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "SPLIT"

train_dataset = MammogramDataset(DATA_DIR / "train", transform=get_transforms(True))
val_dataset = MammogramDataset(DATA_DIR / "val", transform=get_transforms(False))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = get_model().to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

best_val_loss = float("inf")

for epoch in range(EPOCHS):

    # ================= TRAIN =================
    model.train()
    train_loss = 0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

  
    train_loss = train_loss / len(train_loader)

    # ================= VALIDATION =================
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

    # ✅ Divide AFTER loop
    val_loss = val_loss / len(val_loader)

    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pth")
        print("Model saved!")

print("Training complete.")
