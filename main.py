import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

size = 128
batch = 32
epochs = 70
patience = 5

transform = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])

train_data = datasets.ImageFolder(
    "plant_disease/PlantVillage/train", transform=transform
)

val_data = datasets.ImageFolder("plant_disease/PlantVillage/val", transform=transform)

train_loader = DataLoader(train_data, batch_size=batch, shuffle=True)

val_loader = DataLoader(val_data, batch_size=batch, shuffle=False)


class CNN(nn.Module):
    def __init__(self, classes):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 30 * 30, 128),
            nn.ReLU(),
            nn.Linear(128, classes),
        )

    def forward(self, x):
        return self.model(x)


device = "cuda" if torch.cuda.is_available() else "cpu"

model = CNN(len(train_data.classes)).to(device)

loss_fn = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

best_acc = 0
wait = 0

for epoch in range(epochs):
    model.train()

    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        pred = model(x)

        loss = loss_fn(pred, y)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

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

    print(f"epoch {epoch + 1} accuracy {acc:.2f}%")

    if acc > best_acc:
        best_acc = acc
        wait = 0

        torch.save(model.state_dict(), "best_cnn.pth")

    else:
        wait += 1

        if wait >= patience:
            print("early stopping")

            break
