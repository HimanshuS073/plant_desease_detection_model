"""
Plant Disease Classification — Model 1: EfficientNet-B3
========================================================
Architecture : EfficientNet-B3 (CNN, compound scaling)
Strategy     : Transfer learning — freeze early layers, fine-tune last 3
               blocks + custom classifier head
Dataset      : plant_disease/PlantVillage/train  &  .../val
"""

import os
import threading
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# ── Paths ─────────────────────────────────────────────────────────────────────
TRAIN_DIR = "plant_disease/PlantVillage/train"
VAL_DIR = "plant_disease/PlantVillage/val"
SAVE_PATH = "best_efficientnet_b3.pth"

# ── Hyperparameters ───────────────────────────────────────────────────────────
SIZE = 300  # EfficientNet-B3 native resolution
BATCH = 32
EPOCHS = 30
PATIENCE = 7

# ── Device ────────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"device_name:{torch.cuda.get_device_name()}")

# ── Validate paths ────────────────────────────────────────────────────────────
assert os.path.isdir(TRAIN_DIR), f"Train folder not found: '{TRAIN_DIR}'"
assert os.path.isdir(VAL_DIR), f"Val folder not found: '{VAL_DIR}'"

# ── Data Transforms & Loaders ────────────────────────────────────────────────
train_transform = transforms.Compose(
    [
        transforms.Resize((SIZE, SIZE)),
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
        transforms.Resize((SIZE, SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

train_data = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
val_data = datasets.ImageFolder(VAL_DIR, transform=val_transform)

num_classes = len(train_data.classes)
print(f"Number of classes : {num_classes}")
print(f"Train samples     : {len(train_data)}")
print(f"Val samples       : {len(val_data)}")

train_loader = DataLoader(
    train_data, batch_size=BATCH, shuffle=True, num_workers=0, pin_memory=True
)
val_loader = DataLoader(
    val_data, batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=True
)


# ── Visualise Sample Images ───────────────────────────────────────────────────
def imshow(img_tensor, title=None):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = torch.clamp(img_tensor * std + mean, 0, 1).permute(1, 2, 0).numpy()
    plt.imshow(img)
    if title:
        plt.title(title, fontsize=7)
    plt.axis("off")


images, labels = next(iter(train_loader))
fig, axes = plt.subplots(2, 8, figsize=(20, 6))
for i, ax in enumerate(axes.flatten()):
    plt.sca(ax)
    imshow(images[i], train_data.classes[labels[i]])
plt.suptitle("Sample Training Images", fontsize=14)
plt.tight_layout()
plt.savefig("efficientnet_b3_samples.png", dpi=150)
plt.close()

# ── Build EfficientNet-B3 Model ───────────────────────────────────────────────
model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)

# Freeze all parameters first
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the last 3 feature blocks for fine-tuning
for block in list(model.features.children())[-3:]:
    for param in block.parameters():
        param.requires_grad = True

# Replace classifier head
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, num_classes),
)

model = model.to(device)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable params : {trainable:,} / {total:,} ({trainable / total * 100:.1f}%)")

# ── Loss, Optimizer & Scheduler ───────────────────────────────────────────────
# Label smoothing reduces overconfidence and improves generalisation
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

# Two param groups: lower LR for backbone, higher LR for new head
backbone_params = [
    p
    for name, p in model.named_parameters()
    if "classifier" not in name and p.requires_grad
]
classifier_params = [p for name, p in model.named_parameters() if "classifier" in name]

optimizer = optim.AdamW(
    [
        {"params": backbone_params, "lr": 1e-4},
        {"params": classifier_params, "lr": 5e-4},
    ],
    weight_decay=1e-4,
)

# Cosine annealing gives a smoother LR decay
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
print("Optimizer and scheduler ready.")

# ── Per-second watch thread ───────────────────────────────────────────────────
_watch = {
    "epoch":         0,
    "phase":         "idle",
    "batches_done":  0,
    "batches_total": 0,
    "train_correct": 0,
    "train_seen":    0,
    "val_correct":   0,
    "val_seen":      0,
    "loss":          0.0,
    "elapsed":       0.0,
    "running":       True,
}

def watch_thread():
    while _watch["running"]:
        phase   = _watch["phase"]
        epoch   = _watch["epoch"]
        bdone   = _watch["batches_done"]
        btotal  = _watch["batches_total"]
        elapsed = _watch["elapsed"]

        if phase == "train":
            seen = _watch["train_seen"]
            acc  = (_watch["train_correct"] / seen * 100) if seen > 0 else 0.0
            loss = _watch["loss"]
            line = (f"\r[Epoch {epoch:>2}] TRAIN | batch {bdone}/{btotal} | "
                    f"loss {loss:.4f} | train_acc {acc:.2f}% | {elapsed:.0f}s  ")
        elif phase == "val":
            seen = _watch["val_seen"]
            acc  = (_watch["val_correct"] / seen * 100) if seen > 0 else 0.0
            line = (f"\r[Epoch {epoch:>2}]   VAL | batch {bdone}/{btotal} | "
                    f"val_acc {acc:.2f}% | {elapsed:.0f}s  ")
        else:
            time.sleep(1)
            continue

        print(line, end="", flush=True)
        time.sleep(1)

watcher = threading.Thread(target=watch_thread, daemon=True)
watcher.start()

# ── Training Loop ─────────────────────────────────────────────────────────────
history = {"train_loss": [], "val_acc": []}
best_acc = 0.0
wait = 0

for epoch in range(EPOCHS):
    start = time.time()

    # Train
    model.train()
    train_loss    = 0.0
    train_correct = 0
    train_seen    = 0

    _watch.update({
        "epoch": epoch + 1, "phase": "train",
        "batches_done": 0, "batches_total": len(train_loader),
        "train_correct": 0, "train_seen": 0,
        "loss": 0.0, "elapsed": 0.0,
    })

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss   = loss_fn(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss    += loss.item()
        preds          = logits.argmax(dim=1)
        train_correct += (preds == y).sum().item()
        train_seen    += y.size(0)

        _watch["batches_done"]  += 1
        _watch["train_correct"]  = train_correct
        _watch["train_seen"]     = train_seen
        _watch["loss"]           = train_loss
        _watch["elapsed"]        = time.time() - start

    scheduler.step()

    # Validate
    model.eval()
    val_correct = 0
    val_total   = 0

    _watch.update({
        "phase": "val",
        "batches_done": 0, "batches_total": len(val_loader),
        "val_correct": 0, "val_seen": 0,
    })

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            _, predicted = torch.max(model(x), 1)
            val_total   += y.size(0)
            val_correct += (predicted == y).sum().item()

            _watch["batches_done"] += 1
            _watch["val_correct"]   = val_correct
            _watch["val_seen"]      = val_total
            _watch["elapsed"]       = time.time() - start

    _watch["phase"] = "idle"

    acc     = val_correct / val_total * 100
    elapsed = time.time() - start
    history["train_loss"].append(train_loss)
    history["val_acc"].append(acc)

    print(
        f"\n[Epoch {epoch + 1:>2}/{EPOCHS}] "
        f"loss {train_loss:.4f} | "
        f"train_acc {train_correct / train_seen * 100:.2f}% | "
        f"val_acc {acc:.2f}% | "
        f"time {elapsed:.1f}s",
        flush=True,
    )

    if acc > best_acc:
        best_acc = acc
        wait = 0
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"  ✓ Saved → {SAVE_PATH}  (best: {best_acc:.2f}%)", flush=True)
    else:
        wait += 1
        if wait >= PATIENCE:
            print("Early stopping triggered.", flush=True)
            break

_watch["running"] = False
watcher.join(timeout=2)

print(f"\nBest Val Accuracy: {best_acc:.2f}%")

# ── Training Curves ───────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
ax1.plot(history["train_loss"], color="steelblue")
ax1.set_title("Train Loss — EfficientNet-B3")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.grid(True)

ax2.plot(history["val_acc"], color="darkorange")
ax2.set_title("Val Accuracy — EfficientNet-B3")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.grid(True)

plt.tight_layout()
plt.savefig("efficientnet_b3_curves.png", dpi=150)
plt.close()

# ── Per-Class Accuracy ────────────────────────────────────────────────────────
model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
model.eval()

class_correct = [0] * num_classes
class_total = [0] * num_classes

with torch.no_grad():
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        _, predicted = torch.max(model(x), 1)
        for label, pred in zip(y, predicted):
            class_correct[label] += (label == pred).item()
            class_total[label] += 1

print(f"{'Class':<50} {'Correct':>8} {'Total':>7} {'Acc':>7}")
print("-" * 75)
for i, cls in enumerate(train_data.classes):
    acc_i = class_correct[i] / class_total[i] * 100 if class_total[i] > 0 else 0
    print(f"{cls:<50} {class_correct[i]:>8} {class_total[i]:>7} {acc_i:>6.1f}%")
