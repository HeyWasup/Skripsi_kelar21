
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_labels), np.array(all_preds)

def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(title)
    plt.show()

def main_evaluation():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Menggunakan device:", device)
    
    # Definisikan transformasi (sama dengan training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Ukuran standar untuk VGG
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Muat dataset validasi (pastikan struktur folder: dataset/val/<kelas>/...)
    dataset_path = "D:/14. Project Skripsi gw/SimDatatrain"  # Ubah sesuai dengan lokasi validasi Anda
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path '{dataset_path}' tidak ditemukan!")
    
    val_dataset = ImageFolder(root=dataset_path, transform=transform)
    print("Classes:", val_dataset.classes)
    
    batch_size = 32
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Muat model VGG19 yang sudah di-finetune (klasifikasi dua kelas)
    model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).to(device)
    # Bekukan fitur (opsional)
    for param in model.features.parameters():
        param.requires_grad = False
    # Ubah classifier agar output 2 kelas
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, 2)
    model = model.to(device)
    
    # Load model terbaik
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    
    # Evaluasi model pada dataset validasi
    true_labels, preds = evaluate_model(model, val_loader, device)
    
    # Hitung confusion matrix dan classification report
    cm = confusion_matrix(true_labels, preds)
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(true_labels, preds, target_names=val_dataset.classes))
    
    # Visualisasikan confusion matrix
    plot_confusion_matrix(cm, classes=val_dataset.classes, title="Confusion Matrix")

if __name__ == '__main__':
    main_evaluation()