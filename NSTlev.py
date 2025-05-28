import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import copy

# =============================================================================
# 1. Utility Functions: Image Loading & Saving
# =============================================================================
def load_image(image_path, max_size=512, device='cuda'):
    """
    Memuat gambar dari disk, mengubah ukurannya menjadi max_size x max_size,
    dan mengonversinya menjadi tensor.
    """
    transform = transforms.Compose([
        transforms.Resize((max_size, max_size)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # shape: [1, C, H, W]
    return image.to(device, torch.float)

def save_tensor_image(tensor, filename="output.jpg"):
    """
    Menyimpan tensor [1, C, H, W] ke disk sebagai gambar.
    """
    image = tensor.clone().detach().cpu().squeeze(0)
    image = transforms.ToPILImage()(image)
    image.save(filename)

# =============================================================================
# 2. File Selection Using Tkinter File Dialog
# =============================================================================
def select_file(title):
    """
    Membuka dialog file untuk memilih file gambar.
    """
    root = tk.Tk()
    root.withdraw()  # Sembunyikan jendela utama
    file_path = filedialog.askopenfilename(title=title)
    root.destroy()
    return file_path

# =============================================================================
# 3. Load Content (Fractal) and Style (Batik) Images
# =============================================================================
# Pastikan PyTorch menggunakan GPU jika tersedia
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pilih file menggunakan dialog
content_path = select_file("Pilih Gambar Fraktal (Content Image)")
style_path   = select_file("Pilih Gambar Batik (Style Image)")

# Memuat gambar dan mengirimnya ke device
content_img = load_image(content_path, max_size=512, device=device)
style_img   = load_image(style_path,   max_size=512, device=device)

# Inisialisasi gambar generated sebagai salinan dari content image
generated_img = content_img.clone().requires_grad_(True)

# =============================================================================
# 4. Load Pretrained VGG19 untuk Feature Extraction
# =============================================================================
vgg = models.vgg19(pretrained=True).features.to(device).eval()
# Definisikan layer untuk konten dan gaya
content_layers = ['conv4_2']
style_layers   = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

# =============================================================================
# 5. Model Ekstraksi Fitur VGG
# =============================================================================
class VGGFeatureExtractor(nn.Module):
    def __init__(self, style_layers, content_layers):
        super(VGGFeatureExtractor, self).__init__()
        self.style_layers = set(style_layers)
        self.content_layers = set(content_layers)
        self.model = nn.ModuleList()
        for layer in vgg.children():
            self.model.append(layer)
        self.names_map = {}
        block = 1
        conv = 1
        for i, layer in enumerate(self.model):
            if isinstance(layer, nn.Conv2d):
                name = f"conv{block}_{conv}"
                conv += 1
                self.names_map[i] = name
            elif isinstance(layer, nn.ReLU):
                name = f"relu{block}_{conv-1}"
                self.model[i] = nn.ReLU(inplace=False)  # Non in-place untuk menjaga gradien
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

extractor = VGGFeatureExtractor(style_layers, content_layers).to(device)

# =============================================================================
# 6. Compute Target Features: Content and Style (Gram Matrices)
# =============================================================================
def gram_matrix(tensor):
    # tensor shape: [B, C, H, W]
    b, c, h, w = tensor.size()
    features = tensor.view(b * c, h * w)
    G = torch.mm(features, features.t())
    return G.div(b * c * h * w)

with torch.no_grad():
    style_feats, _ = extractor(style_img)
    _, content_feats_ref = extractor(content_img)

# Hitung Gram matrix target untuk style
style_grams_ref = {}
for layer_name, feat_map in style_feats.items():
    style_grams_ref[layer_name] = gram_matrix(feat_map)

# Bobot loss
content_weight = 1
style_weight = 1e6

# =============================================================================
# 7. Define Optimizer and Loss Function
# =============================================================================
optimizer = optim.LBFGS([generated_img])
max_steps = 300
iteration = [0]

def closure():
    with torch.no_grad():
        generated_img.clamp_(0, 1)  # Pastikan nilai piksel di [0,1]
    optimizer.zero_grad()
    style_out, content_out = extractor(generated_img)
    
    # Hitung Content Loss
    c_loss = 0
    for cl_name in content_out:
        c_loss += nn.functional.mse_loss(content_out[cl_name], content_feats_ref[cl_name])
    
    # Hitung Style Loss
    s_loss = 0
    for st_name in style_out:
        G = gram_matrix(style_out[st_name])
        A = style_grams_ref[st_name]
        s_loss += nn.functional.mse_loss(G, A)
    
    total_loss = content_weight * c_loss + style_weight * s_loss
    total_loss.backward()
    
    if iteration[0] % 50 == 0:
        print(f"Iteration {iteration[0]}/{max_steps}, Total loss: {total_loss.item():.4f} "
              f"(Content: {c_loss.item():.4f}, Style: {s_loss.item():.4f})")
    iteration[0] += 1
    return total_loss

for step in range(max_steps):
    optimizer.step(closure)
    with torch.no_grad():
        generated_img.clamp_(0, 1)

# =============================================================================
# 8. Save the Final Image
# =============================================================================
save_tensor_image(generated_img, "fractal_stylized_output3.jpg")
print("Style transfer completed and image saved as 'fractal_stylized_output3.jpg'.")