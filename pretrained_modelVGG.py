import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from PIL import Image, UnidentifiedImageError
import copy

# Custom PIL loader yang menangani error saat membuka gambar
def pil_loader(path):
    try:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
    except UnidentifiedImageError as e:
        print(f"Warning: Gagal membuka {path}. Error: {e}. Mengembalikan gambar hitam.")
        # Mengembalikan gambar hitam dengan ukuran standar (224x224)
        return Image.new('RGB', (224, 224), (0, 0, 0))
    except Exception as e:
        print(f"Warning: Terjadi error saat membuka {path}. Error: {e}. Mengembalikan gambar hitam.")
        return Image.new('RGB', (224, 224), (0, 0, 0))

# =============================================================================
# 1. Definisikan Transformasi Data
# =============================================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Ukuran standar untuk VGG
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# =============================================================================
# 2. Muat Dataset Menggunakan ImageFolder dengan Custom Loader
# =============================================================================
dataset_path = "D:/14. Project Skripsi gw/SimDatatrain"  # Ganti dengan path dataset Anda
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset path '{dataset_path}' tidak ditemukan!")

# Gunakan custom loader dengan parameter loader
dataset = ImageFolder(root=dataset_path, transform=transform, loader=pil_loader)
print("Classes:", dataset.classes)

# Bagi dataset menjadi training (80%) dan validasi (20%)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

batch_size = 32
# Jika terjadi masalah multiprocessing di Windows, set num_workers=0
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# =============================================================================
# 3. Buat Model Klasifikasi dengan VGG19 Pretrained
# =============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Menggunakan device:", device)
model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).to(device)

# Opsional: Bekukan parameter fitur untuk mempercepat training
for param in model.features.parameters():
    param.requires_grad = False

# Ubah classifier untuk output dua kelas (misalnya, 'batikmegamendung' dan 'fraktalsampai50')
num_features = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_features, 2)
model = model.to(device)

# =============================================================================
# 4. Definisikan Loss Function dan Optimizer
# =============================================================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

# =============================================================================
# 5. Training Loop
# =============================================================================
num_epochs = 10  # Ubah sesuai kebutuhan
best_val_acc = 0.0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}")
    
    # Evaluasi pada set validasi
    model.eval()
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
    val_acc = correct_val / total_val
    print(f"Validation Accuracy: {val_acc:.4f}")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")
        print("Model saved!")

print("Training completed.")