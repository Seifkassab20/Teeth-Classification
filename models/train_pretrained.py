import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import json

from dataloader.dataloader import get_dataloaders
from models.pretrained import PretrainedModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_path = r"Teeth DataSet\Teeth_Dataset"
if not os.path.exists(dataset_path):
    print(f"Dataset path not found: {dataset_path}")
    exit()

train_loader, val_loader, _, classes = get_dataloaders(dataset_path)

with open("class_names.json", "w") as f:
    json.dump((classes), f)
    print("Class names saved to class_names.json")
num_classes = len(classes)

model = PretrainedModel(num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4 , weight_decay=1e-4)

EPOCHS = 20

for epoch in range(EPOCHS):

    model.train()
    correct, total = 0, 0
    running_loss = 0.0

    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total
    train_loss = running_loss / len(train_loader)


    model.eval()
    correct_val, total_val = 0, 0
    val_loss = 0.0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct_val += (preds == labels).sum().item()
            total_val += labels.size(0)

    val_acc = correct_val / total_val
    val_loss = val_loss / len(val_loader)

    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

torch.save(model.state_dict(), "weights/medical_pretrained.pth")
print("Medical pretrained weights saved")
