import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import copy
import tkinter as tk
from tkinter import filedialog

# Pilih GPU jika tersedia, jika tidak maka gunakan CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ukuran gambar yang akan diproses (512 atau 800)
imsize = 800

# Transformasi untuk memuat dan mengubah ukuran gambar
loader = transforms.Compose([
    transforms.Resize((imsize, imsize)),
    transforms.ToTensor()
])

def load_image(image_path):
    """Memuat gambar dan mengubahnya menjadi tensor dengan ukuran (imsize x imsize)."""
    image = Image.open(image_path)
    image = loader(image).unsqueeze(0)  # Tambahkan batch dimension
    return image.to(device, torch.float)

# Untuk mengubah tensor menjadi gambar PIL
unloader = transforms.ToPILImage()

# Normalisasi yang digunakan oleh VGG19 (berdasarkan mean dan std ImageNet)
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)
        
    def forward(self, img):
        return (img - self.mean) / self.std

class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
        
    def forward(self, input):
        self.loss = nn.functional.mse_loss(input, self.target)
        return input

def gram_matrix(input):
    """Menghitung Gram Matrix untuk representasi gaya."""
    a, b, c, d = input.size()  # a=batch size, b=feature maps, c=dims
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()
        
    def forward(self, input):
        G = gram_matrix(input)
        self.loss = nn.functional.mse_loss(G, self.target)
        return input

# Layer yang sering dipakai untuk konten & gaya (bisa dimodifikasi)
content_layers_default = ['conv_4']
style_layers_default   = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    """
    Membangun model 'gabungan' dengan penambahan ContentLoss & StyleLoss
    di layer-layer tertentu.
    """
    cnn = copy.deepcopy(cnn)
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    content_losses = []
    style_losses = []
    
    model = nn.Sequential(normalization)

    i = 0  # counter konvolusi
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f"conv_{i}"
        elif isinstance(layer, nn.ReLU):
            name = f"relu_{i}"
            # ubah ReLU inplace=False agar gradient tidak hilang
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f"pool_{i}"
        elif isinstance(layer, nn.BatchNorm2d):
            name = f"bn_{i}"
        else:
            raise RuntimeError(f"Layer tidak dikenali: {layer.__class__.__name__}")

        model.add_module(name, layer)

        if name in content_layers:
            # Hitung target fitur konten
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # Hitung target fitur gaya
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

    # Potong model setelah layer terakhir yang mengandung loss
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break
    model = model[:(i + 1)]
    
    return model, style_losses, content_losses

def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, 
                       num_steps=300, style_weight=1e6, content_weight=1):
    """
    Proses style transfer: optimisasi gambar input agar menyesuaikan
    konten dan gaya.
    """
    print('Mulai optimisasi...')
    model, style_losses, content_losses = get_style_model_and_losses(
        cnn, normalization_mean, normalization_std,
        style_img, content_img
    )
    
    # Gunakan L-BFGS
    optimizer = optim.LBFGS([input_img.requires_grad_()])

    run = [0]
    while run[0] <= num_steps:
        def closure():
            # Pastikan nilai piksel tetap [0..1]
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            
            style_score = 0.0
            content_score = 0.0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            loss = style_weight * style_score + content_weight * content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print(f"Step {run[0]}: Style Loss = {style_score.item():.4f}, "
                      f"Content Loss = {content_score.item():.4f}")
            return loss

        optimizer.step(closure)
    
    # Normalisasi hasil akhir
    input_img.data.clamp_(0, 1)
    return input_img

def get_file_path(prompt="Pilih file"):
    """
    Membuka dialog file chooser untuk memilih file gambar.
    """
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(title=prompt)
    root.destroy()
    return path

def main():
    # Pilih gambar fractal (content)
    fractal_content_path = get_file_path("Pilih gambar fractal (Content)")
    if not fractal_content_path:
        print("Tidak ada file fractal yang dipilih. Keluar.")
        return
    
    # Pilih gambar batik (style)
    batik_style_path = get_file_path("Pilih gambar batik (Style)")
    if not batik_style_path:
        print("Tidak ada file style yang dipilih. Keluar.")
        return

    # Load content & style images
    content_img = load_image(fractal_content_path)
    style_img = load_image(batik_style_path)

    # Pastikan keduanya berukuran sama (sudah di-resize dengan transform di atas)
    assert content_img.size() == style_img.size(), \
        "Kedua gambar harus memiliki ukuran yang sama!"

    # Inisialisasi gambar input (bisa copy dari content_img)
    input_img = content_img.clone()

    # Load model VGG19 pretrained (hanya feature extractor)
    cnn = models.vgg19(pretrained=True).features.to(device).eval()

    # Jalankan style transfer
    output = run_style_transfer(
        cnn, cnn_normalization_mean, cnn_normalization_std,
        content_img, style_img, input_img,
        num_steps=300,         # Jumlah iterasi
        style_weight=1e6,      # Bobot untuk gaya
        content_weight=1       # Bobot untuk konten
    )

    # Konversi hasil ke format PIL Image
    output_image = output.cpu().clone().squeeze(0)
    output_image = unloader(output_image)

    # Simpan hasil
    output_image.save("hasil_style_transfer2.jpg")
    print("Hasil style transfer disimpan sebagai 'hasil_style_transfer2.jpg'")

if __name__ == "__main__":
    main()