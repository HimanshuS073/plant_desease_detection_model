import torch
import torch.nn as nn
from PIL import Image
from torchvision import datasets, transforms


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

train_data = datasets.ImageFolder("plant_disease/PlantVillage/train")
classes = train_data.classes

model = CNN(len(classes)).to(device)
model.load_state_dict(torch.load("best_cnn.pth", map_location=device))
model.eval()

t = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])

img_path = "plant_disease/PlantVillage/val/Tomato___Tomato_Yellow_Leaf_Curl_Virus/0a1d1def-462c-46d3-90e6-2a11fcb45a21___UF.GRC_YLCV_Lab 01675.JPG"

img = Image.open(img_path).convert("RGB")
x = t(img).unsqueeze(0).to(device)

with torch.no_grad():
    out = model(x)
    pred = torch.argmax(out, 1).item()

print(classes[pred])
