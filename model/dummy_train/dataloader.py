import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import numpy as np

class BDDDataset(Dataset):
    """
    Custom Dataset for loading images and labels in YOLO format from the BDD dataset.
    """
    def __init__(self, img_dir, label_dir, img_size=640):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        # 1. Load Image
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. Resize and Normalize
        image = cv2.resize(image, (self.img_size, self.img_size))
        image = image.transpose(2, 0, 1) / 255.0  # HWC to CHW

        # 3. Load Labels (YOLO format: class, x, y, w, h)
        label_path = os.path.join(self.label_dir, self.img_files[idx].replace('.jpg', '.txt'))
        if os.path.exists(label_path):
            labels = np.loadtxt(label_path).reshape(-1, 5)
        else:
            labels = np.zeros((0, 5))  # Empty if no objects

        return torch.from_numpy(image).float(), torch.from_numpy(labels).float()


# Collate function to handle varying number of objects per image
def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images, dim=0)
    # Pad labels to the same length and stack, or concatenate with image indices
    max_objs = max(l.shape[0] for l in labels)
    if max_objs == 0:
        padded_labels = torch.zeros(len(labels), 0, 5)
    else:
        padded_labels = torch.zeros(len(labels), max_objs, 5)
        for i, l in enumerate(labels):
            if l.shape[0] > 0:
                padded_labels[i, :l.shape[0], :] = l
    return images, padded_labels
