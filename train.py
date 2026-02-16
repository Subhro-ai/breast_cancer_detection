import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from pathlib import Path

from src.dataset import MammogramDataset
from src.transform import get_transforms
from src.model import get_model
from src.loss import FocalLoss

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

BATCH_SIZE = 8
EPOCHS = 35
LR = 1e-4
FOLDS = 5

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "FINAL" / "CBIS-DDSM"

# Load full dataset (no split folder now)
full_dataset = MammogramDataset(DATA_DIR, transform=get_transforms(True))

labels = np.array(full_dataset.labels)

skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

fold_results = []

for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
    print(f"\n========== Fold {fold+1} ==========")

    train_subset = Subset(full_dataset, train_idx)
    val_subset = Subset(full_dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

    model = get_model().to(DEVICE)

    criterion = FocalLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=4, factor=0.5
    )

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        # TRAIN
        model.train()
        train_loss = 0

        for images, labels_batch in train_loader:
            images = images.to(DEVICE)
            labels_batch = labels_batch.to(DEVICE).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # VALIDATE
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for images, labels_batch in val_loader:
                images = images.to(DEVICE)
                labels_batch = labels_batch.to(DEVICE).unsqueeze(1)

                outputs = model(images)
                loss = criterion(outputs, labels_batch)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")

    fold_results.append(val_loss)

print("\n===== CROSS VALIDATION RESULTS =====")
print("Average Validation Loss:", np.mean(fold_results))
