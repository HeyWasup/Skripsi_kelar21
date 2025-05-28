import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import copy
import matplotlib.pyplot as plt

# ------------------------
# 1. Image Loading & Utility
# ------------------------
def load_image(image_path, max_size=512, device='cuda'):
    transform = transforms.Compose([
        transforms.Resize((max_size, max_size)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert("RGB")  # Pastikan gambar hanya 3 channel
    image = transform(image).unsqueeze(0)  # shape: [1, C, H, W]
    return image.to(device, torch.float)

def save_tensor_image(tensor, filename="nst_output.jpg"):
    image = tensor.clone().detach().cpu().squeeze(0)
    image = transforms.ToPILImage()(image)
    image.save(filename)

def select_file(title):
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title=title)
    root.destroy()
    return file_path

# ------------------------
# 2. Load Input Images untuk NST
# ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Menggunakan device:", device)

content_path = select_file("Pilih Gambar Fraktal (Content Image)")
style_path = select_file("Pilih Gambar Batik (Style Image)")

content_img = load_image(content_path, max_size=512, device=device)
style_img = load_image(style_path, max_size=512, device=device)
generated_img = content_img.clone().requires_grad_(True)

# ------------------------
# 3. Load VGG19 Fine-Tuned (Model Terbaik) untuk Ekstraksi Fitur
# ------------------------
# Muat model VGG19 yang sudah di-finetune (klasifikasi batik vs. fraktal)
model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).to(device)
# Bekukan parameter fitur jika diinginkan
for param in model.features.parameters():
    param.requires_grad = False
# Ubah classifier agar output 2 kelas
num_features = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_features, 2)
model = model.to(device)
# Muat state dict model terbaik (pastikan path sudah benar)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

# Gunakan bagian features dari model terbaik sebagai feature extractor untuk NST
fine_tuned_extractor = model.features

# Definisikan layer yang akan digunakan untuk content dan style
content_layers = ['conv4_2']
style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

# ------------------------
# 4. Build VGGFeatureExtractor Menggunakan fine_tuned_extractor
# ------------------------
class VGGFeatureExtractor(nn.Module):
    def __init__(self, style_layers, content_layers, extractor):
        super(VGGFeatureExtractor, self).__init__()
        self.style_layers = set(style_layers)
        self.content_layers = set(content_layers)
        # Gunakan feature extractor dari model fine-tuned
        self.model = extractor  
        self.names_map = {}
        block = 1
        conv = 1
        # Mapping layer: perlu disesuaikan dengan struktur model.features
        for i, layer in enumerate(self.model):
            if isinstance(layer, nn.Conv2d):
                name = f"conv{block}_{conv}"
                conv += 1
                self.names_map[i] = name
            elif isinstance(layer, nn.ReLU):
                name = f"relu{block}_{conv-1}"
                self.names_map[i] = name
            elif isinstance(layer, nn.MaxPool2d):
                name = f"pool{block}"
                block += 1
                conv = 1
                self.names_map[i] = name

    def forward(self, x):
        style_outputs = {}
        content_outputs = {}
        for i, layer in enumerate(self.model):
            x = layer(x)
            name = self.names_map.get(i, None)
            if name in self.style_layers:
                style_outputs[name] = x
            if name in self.content_layers:
                content_outputs[name] = x
        return style_outputs, content_outputs

extractor_ft = VGGFeatureExtractor(style_layers, content_layers, fine_tuned_extractor).to(device)

# ------------------------
# 5. Compute Style & Content Targets
# ------------------------
def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    features = tensor.view(b * c, h * w)
    G = torch.mm(features, features.t())
    return G.div(b * c * h * w)

with torch.no_grad():
    style_feats, _ = extractor_ft(style_img)
    _, content_feats_ref = extractor_ft(content_img)

style_grams_ref = {}
for layer_name, feat_map in style_feats.items():
    style_grams_ref[layer_name] = gram_matrix(feat_map)

content_weight = 1
style_weight = 1e6

# ------------------------
# 6. Define Optimizer & Loss Function untuk NST
# ------------------------
optimizer = optim.LBFGS([generated_img])
max_steps = 300
iteration = [0]

def closure():
    with torch.no_grad():
        generated_img.clamp_(0, 1)
    optimizer.zero_grad()
    style_out, content_out = extractor_ft(generated_img)
    c_loss = 0
    for cl_name in content_out:
        c_loss += nn.functional.mse_loss(content_out[cl_name], content_feats_ref[cl_name])
    s_loss = 0
    for st_name in style_out:
        G = gram_matrix(style_out[st_name])
        A = style_grams_ref[st_name]
        s_loss += nn.functional.mse_loss(G, A)
    total_loss = content_weight * c_loss + style_weight * s_loss
    total_loss.backward()
    if iteration[0] % 50 == 0:
        print(f"Iteration {iteration[0]}/{max_steps}, Total loss: {total_loss.item():.4f} (Content: {c_loss.item():.4f}, Style: {s_loss.item():.4f})")
    iteration[0] += 1
    return total_loss

for i in range(max_steps):
    optimizer.step(closure)
    with torch.no_grad():
        generated_img.clamp_(0, 1)

# ------------------------
# 7. Save the Final NST Output
# ------------------------
save_tensor_image(generated_img, "nst_with_finetuned_vgg.jpg")
print("NST completed and image saved as 'nst_with_finetuned_vgg.jpg'.")
