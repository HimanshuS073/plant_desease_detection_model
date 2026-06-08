import threading
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# ── Hyperparameters ──────────────────────────────────────────────────────────
SIZE     = 224
BATCH    = 32
EPOCHS   = 30
PATIENCE = 7

# ── Transforms ───────────────────────────────────────────────────────────────
train_transform = transforms.Compose([
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
])

val_transform = transforms.Compose([
    transforms.Resize((SIZE, SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── Data ─────────────────────────────────────────────────────────────────────
train_data = datasets.ImageFolder("plant_disease/PlantVillage/train", transform=train_transform)
val_data   = datasets.ImageFolder("plant_disease/PlantVillage/val",   transform=val_transform)

num_classes = len(train_data.classes)
print(f"classes: {num_classes}", flush=True)

train_loader = DataLoader(train_data, batch_size=BATCH, shuffle=True,  num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_data,   batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=True)

# ── Device ───────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}", flush=True)

# ── Model: EfficientNet-B3 ────────────────────────────────────────────────────
model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)

for param in model.parameters():
    param.requires_grad = False

for block in list(model.features.children())[-3:]:
    for param in block.parameters():
        param.requires_grad = True

in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, num_classes),
)

model = model.to(device)

# ── Loss & Optimizer ─────────────────────────────────────────────────────────
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

backbone_params   = [p for name, p in model.named_parameters()
                     if "classifier" not in name and p.requires_grad]
classifier_params = [p for name, p in model.named_parameters()
                     if "classifier" in name]

optimizer = optim.AdamW([
    {"params": backbone_params,   "lr": 1e-4},
    {"params": classifier_params, "lr": 5e-4},
], weight_decay=1e-4)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

# ── Per-second watch thread ───────────────────────────────────────────────────
# Shared state dict — main thread writes, watch thread reads
_watch = {
    "epoch":        0,
    "phase":        "idle",   # "train" | "val"
    "batches_done": 0,
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
    """Prints a live status line every second using \\r (overwrites itself)."""
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
            bar  = f"batch {bdone}/{btotal}"
            line = (f"\r[Epoch {epoch:>2}] TRAIN | {bar} | "
                    f"loss {loss:.4f} | train_acc {acc:.2f}% | {elapsed:.0f}s  ")
        elif phase == "val":
            seen = _watch["val_seen"]
            acc  = (_watch["val_correct"] / seen * 100) if seen > 0 else 0.0
            bar  = f"batch {bdone}/{btotal}"
            line = (f"\r[Epoch {epoch:>2}]   VAL | {bar} | "
                    f"val_acc {acc:.2f}% | {elapsed:.0f}s  ")
        else:
            time.sleep(1)
            continue

        print(line, end="", flush=True)
        time.sleep(1)

watcher = threading.Thread(target=watch_thread, daemon=True)
watcher.start()

# ── Training loop ─────────────────────────────────────────────────────────────
best_acc = 0.0
wait     = 0

for epoch in range(EPOCHS):
    epoch_start = time.time()

    # ── Train ─────────────────────────────────────────────────────────────────
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

        train_loss += loss.item()
        preds       = logits.argmax(dim=1)
        train_correct += (preds == y).sum().item()
        train_seen    += y.size(0)

        _watch["batches_done"]  += 1
        _watch["train_correct"]  = train_correct
        _watch["train_seen"]     = train_seen
        _watch["loss"]           = train_loss
        _watch["elapsed"]        = time.time() - epoch_start

    scheduler.step()

    # ── Validate ──────────────────────────────────────────────────────────────
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
            _watch["elapsed"]       = time.time() - epoch_start

    _watch["phase"] = "idle"

    acc     = val_correct / val_total * 100
    elapsed = time.time() - epoch_start
    current_lr = optimizer.param_groups[1]["lr"]

    # newline to end the \\r line, then print epoch summary
    print(
        f"\n[Epoch {epoch + 1:>2}/{EPOCHS}] "
        f"loss {train_loss:.4f} | "
        f"train_acc {train_correct / train_seen * 100:.2f}% | "
        f"val_acc {acc:.2f}% | "
        f"lr {current_lr:.2e} | "
        f"time {elapsed:.1f}s",
        flush=True,
    )

    # ── Checkpoint & early stop ───────────────────────────────────────────────
    if acc > best_acc:
        best_acc = acc
        wait     = 0
        torch.save(model.state_dict(), "best_cnn.pth")
        print(f"  ✓ model saved (best val_acc {best_acc:.2f}%)", flush=True)
    else:
        wait += 1
        if wait >= PATIENCE:
            print("early stopping triggered", flush=True)
            break

_watch["running"] = False
watcher.join(timeout=2)

print(f"\nbest val accuracy: {best_acc:.2f}%", flush=True)
