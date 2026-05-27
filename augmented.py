import os
import random

import torch
from PIL import Image, ImageFilter
from torchvision import transforms

root_dir = "plant_disease/PlantVillage/train"

augment = transforms.Compose(
    [
        transforms.RandomRotation(40),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(
            brightness=0.5,
            contrast=0.5,
            saturation=0.5,
            hue=0.2,
        ),
    ]
)

for root, dirs, files in os.walk(root_dir):
    for file in files:
        if "_aug_" in file:
            continue

        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        path = os.path.join(root, file)

        try:
            img = Image.open(path).convert("RGB")

        except Exception as e:
            print(e)
            continue

        name = os.path.splitext(file)[0]

        print(f"processing {path}")

        for i in range(50):
            x = augment(img)

            if random.random() < 0.7:
                x = x.filter(ImageFilter.GaussianBlur(radius=random.uniform(1, 4)))

            if random.random() < 0.5:
                t = transforms.ToTensor()(x)

                t = t + torch.randn_like(t) * 0.08

                t = torch.clamp(t, 0, 1)

                x = transforms.ToPILImage()(t)

            if random.random() < 0.5:
                w, h = x.size

                left = random.randint(0, int(w * 0.2))
                top = random.randint(0, int(h * 0.2))

                right = random.randint(
                    int(w * 0.8),
                    w,
                )

                bottom = random.randint(
                    int(h * 0.8),
                    h,
                )

                x = x.crop(
                    (
                        left,
                        top,
                        right,
                        bottom,
                    )
                )

                x = x.resize((224, 224))

            save_path = os.path.join(
                root,
                f"{name}_aug_{i}.jpg",
            )

            x.save(save_path)

            print(save_path)

print("done")
