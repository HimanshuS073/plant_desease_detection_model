import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

size = 224
batch = 32
epochs = 20
patience = 5

train_transform = transforms.Compose(
    [
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(25),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.8, 1.2)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.GaussianBlur(kernel_size=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

train_data = datasets.ImageFolder(
    "plant_disease/PlantVillage/train", transform=train_transform
)

val_data = datasets.ImageFolder(
    "plant_disease/PlantVillage/val", transform=val_transform
)

train_loader = DataLoader(
    train_data, batch_size=batch, shuffle=True, num_workers=4, pin_memory=True
)

val_loader = DataLoader(
    val_data, batch_size=batch, shuffle=False, num_workers=4, pin_memory=True
)

device = "cuda" if torch.cuda.is_available() else "cpu"

print(device)

model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

for param in model.features.parameters():
    param.requires_grad = False

model.classifier = nn.Sequential(
    nn.Dropout(0.5), nn.Linear(model.classifier[1].in_features, len(train_data.classes))
)

model = model.to(device)

loss_fn = nn.CrossEntropyLoss()

optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", patience=2, factor=0.5
)

best_acc = 0
wait = 0

for epoch in range(epochs):
    start = time.time()

    model.train()

    train_loss = 0

    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        pred = model(x)

        loss = loss_fn(pred, y)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)

            _, predicted = torch.max(pred, 1)

            total += y.size(0)

            correct += (predicted == y).sum().item()

    acc = correct / total * 100

    scheduler.step(acc)

    end = time.time()

    print(
        f"epoch {epoch + 1} "
        f"loss {train_loss:.4f} "
        f"acc {acc:.2f}% "
        f"time {end - start:.2f}s"
    )

    if acc > best_acc:
        best_acc = acc
        wait = 0

        torch.save(model.state_dict(), "best_model.pth")

        print("model saved")

    else:
        wait += 1

        if wait >= patience:
            print("early stopping")

            break

print(f"best accuracy {best_acc:.2f}%")
