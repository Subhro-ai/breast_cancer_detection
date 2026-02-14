from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


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

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("L")  # grayscale
        image = image.convert("RGB")  # convert to 3-channel for pretrained model

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)
