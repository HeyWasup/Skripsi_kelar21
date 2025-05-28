import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

def generate_mandelbrot(x_min, x_max, y_min, y_max, width, height, max_iter, device):
    """
    Menghasilkan array 2D berisi nilai iterasi Mandelbrot untuk setiap piksel menggunakan PyTorch.
    
    Parameter:
      - x_min, x_max: Batas sumbu X (real)
      - y_min, y_max: Batas sumbu Y (imajiner)
      - width, height: Resolusi gambar
      - max_iter: Jumlah iterasi maksimum per piksel
      - device: device PyTorch (GPU atau CPU)
      
    Mengembalikan:
      Array 2D (numpy array) dengan nilai iterasi tiap piksel.
    """
    # Buat koordinat secara linier dengan PyTorch
    x = torch.linspace(x_min, x_max, width, device=device)
    y = torch.linspace(y_min, y_max, height, device=device)
    # Gunakan meshgrid; perhatikan urutan (y, x) agar bentuk array (height, width)
    Y, X = torch.meshgrid(y, x, indexing='ij')
    # Buat bilangan kompleks
    C = X + 1j * Y  
    Z = torch.zeros_like(C, device=device)
    M = torch.full(C.shape, max_iter, device=device, dtype=torch.int32)

    for i in range(max_iter):
        mask = torch.abs(Z) <= 2
        Z[mask] = Z[mask]**2 + C[mask]
        diverged = mask & (torch.abs(Z) > 2)
        M[diverged] = i
    return M.cpu().numpy()

# Set device untuk PyTorch (GPU jika tersedia)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Menggunakan device:", device)

# Buat folder output untuk menyimpan gambar
output_folder = "MandelbrotOutputs"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Gunakan font default untuk anotasi teks
font = ImageFont.load_default()

# Loop untuk menghasilkan 500 gambar fraktal
for i in range(1000):
    # Variasikan parameter secara bertahap:
    max_iter = 50 + i            # max_iter mulai dari 50 hingga 
    zoom = 1 + i * 0.05          # Faktor zoom meningkat secara linear
    center_x, center_y = -0.75, 0.0
    scale = 1.5 / zoom           # Luas area tampilan mengecil seiring zoom

    # Hitung batas wilayah kompleks
    x_min = center_x - scale
    x_max = center_x + scale
    y_min = center_y - scale
    y_max = center_y + scale

    # Tentukan resolusi gambar
    width, height = 512 , 512

    # Generate fraktal Mandelbrot menggunakan GPU (jika tersedia)
    fractal = generate_mandelbrot(x_min, x_max, y_min, y_max, width, height, max_iter, device)

    # Normalisasi array ke rentang 0-255 untuk citra grayscale
    norm_fractal = (fractal.astype(np.float32) / fractal.max()) * 255
    norm_fractal = norm_fractal.astype(np.uint8)
    img = Image.fromarray(norm_fractal, mode='L')

    # Ubah ke mode RGB agar dapat ditambahkan teks berwarna
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)

    # Siapkan keterangan parameter yang akan ditampilkan
    text = (f"Output: {i+1}/500\n"
            f"max_iter: {max_iter}\n"
            f"zoom: {zoom:.2f}\n"
            f"center: ({center_x}, {center_y})\n"
            f"scale: {scale:.3f}")
    draw.text((10, 10), text, fill=(255, 255, 255), font=font)

    # Simpan gambar dengan nama berurutan, misalnya: mandelbrot_000.png, mandelbrot_001.png, dst.
    filename = os.path.join(output_folder, f"mandelbrot_{i:03d}.jpg")
    img.save(filename)
    print(f"Saved {filename}")