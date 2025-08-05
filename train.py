import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from utils.model import ResNet9  # Make sure your model.py matches!
from torch.utils.data import DataLoader

# ===============================
# 1️⃣ CONFIG
# ===============================
data_dir = 'data/PlantVillage'  # <-- Put your PlantVillage dataset here!
num_classes = 3  # Change if you have more
batch_size = 32
num_epochs = 10
learning_rate = 0.001

# ===============================
# 2️⃣ TRANSFORMS
# ===============================
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# ===============================
# 3️⃣ DATASET + LOADER
# ===============================
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

print(f"Number of classes found: {len(dataset.classes)}")

# ===============================
# 4️⃣ MODEL
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet9(3, len(dataset.classes)).to(device)

# ===============================
# 5️⃣ LOSS + OPTIMIZER
# ===============================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ===============================
# 6️⃣ TRAIN LOOP
# ===============================
for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

print("✅ Training done!")

# ===============================
# 7️⃣ SAVE
# ===============================
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/plant_disease_model.pth")
print("✅ Model saved to models/plant_disease_model.pth")
