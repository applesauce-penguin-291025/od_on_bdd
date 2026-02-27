from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from model.dummy_train.dataloader import BDDDataset, collate_fn
from model.dummy_train.model import DummyDetector

#Training on CPU for simplicity. For real training, GPU will be used if available.
device = torch.device('cpu')
#Use torch.device("cuda" if torch.cuda.is_available() else "cpu") for training on GPU
print(f"Using device: {device}")

IMG_DIR = "bdd_yolo/train/images"
LABEL_DIR = "bdd_yolo/train/labels"

dataset = BDDDataset(
    img_dir=IMG_DIR,
    label_dir=LABEL_DIR
)

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=collate_fn
)

model = DummyDetector().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

model.train()
for batch_idx, (imgs, targets) in enumerate(loader):
    imgs = imgs.to(device)

    preds = model(imgs)

    if targets.numel() == 0:
        continue

    # Very simplified loss for this demo
    loss = loss_fn(preds[:, 1:], targets[:preds.shape[0], 0, 1:].to(device))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if batch_idx % 50 == 0:
        print(f"Batch {batch_idx}/{len(loader)}, Loss: {loss.item():.4f}")

print("Epoch complete!")
