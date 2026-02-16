from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2
import numpy as np


class MammogramDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform

        self.classes = ["BENIGN", "MALIGNANT"]
        self.image_paths = []
        self.labels = []

        for label, class_name in enumerate(self.classes):
            class_dir = self.root_dir / class_name
            for img_path in class_dir.glob("*.jpg"):
                self.image_paths.append(img_path)
                self.labels.append(label)

        # CLAHE setup
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def apply_clahe(self, img):
        img = np.array(img)
        img = self.clahe.apply(img)
        return Image.fromarray(img)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("L")
        image = self.apply_clahe(image)
        image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)
