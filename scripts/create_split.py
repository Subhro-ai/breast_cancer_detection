import os
import shutil
import random
from pathlib import Path
from sklearn.model_selection import train_test_split

# ===============================
# CONFIG
# ===============================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "FINAL" / "CBIS-DDSM"
SPLIT_DIR = BASE_DIR / "data" / "SPLIT"

random.seed(42)

# Create split directories
for split in ["train", "val", "test"]:
    for label in ["BENIGN", "MALIGNANT"]:
        (SPLIT_DIR / split / label).mkdir(parents=True, exist_ok=True)

# ===============================
# PROCESS EACH CLASS
# ===============================
for label in ["BENIGN", "MALIGNANT"]:
    images = list((DATA_DIR / label).glob("*.jpg"))
    images = [str(p) for p in images]

    # First split: Train (70%) and Temp (30%)
    train_imgs, temp_imgs = train_test_split(
        images, test_size=0.30, random_state=42
    )

    # Second split: Validation (15%) and Test (15%)
    val_imgs, test_imgs = train_test_split(
        temp_imgs, test_size=0.50, random_state=42
    )

    splits = {
        "train": train_imgs,
        "val": val_imgs,
        "test": test_imgs,
    }

    # Copy files
    for split_name, file_list in splits.items():
        for file_path in file_list:
            dst = SPLIT_DIR / split_name / label / Path(file_path).name
            shutil.copy(file_path, dst)

print("Dataset split complete.")
