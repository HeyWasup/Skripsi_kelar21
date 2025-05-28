import torch
import numpy as np
from PIL import Image
import os

def generate_mandelbrot(x_min, x_max, y_min, y_max, width, height, max_iter, device, angle=0):
    """
    Menghasilkan array 2D berisi nilai iterasi Mandelbrot untuk setiap piksel menggunakan PyTorch,
    dengan rotasi tambahan terhadap koordinat.
    
    Parameter:
      - x_min, x_max: Batas sumbu X (real)
      - y_min, y_max: Batas sumbu Y (imajiner)
      - width, height: Resolusi gambar
      - max_iter: Jumlah iterasi maksimum per piksel
      - device: device PyTorch (GPU atau CPU)
      - angle: sudut rotasi (radian) yang diterapkan pada koordinat
       
    Mengembalikan:
      Array 2D (numpy array) dengan nilai iterasi tiap piksel.
    """
    # Buat koordinat secara linier dengan PyTorch
    x = torch.linspace(x_min, x_max, width, device=device)
    y = torch.linspace(y_min, y_max, height, device=device)
    # Buat meshgrid; urutan (y, x) menghasilkan tensor berukuran (height, width)
    Y, X = torch.meshgrid(y, x, indexing='ij')
    
    # Terapkan rotasi pada koordinat
    cos_angle = torch.cos(torch.tensor(angle, device=device))
    sin_angle = torch.sin(torch.tensor(angle, device=device))
    X_rot = X * cos_angle - Y * sin_angle
    Y_rot = X * sin_angle + Y * cos_angle
    
    # Buat bilangan kompleks dari koordinat yang sudah diputar
    C = X_rot + 1j * Y_rot  
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
output_folder = "MandelbrotOutputs(add Rotation,512res&no text)"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop untuk menghasilkan 1000 gambar fraktal
for i in range(1000):
    # Variasikan parameter secara bertahap:
    max_iter = 50 + i            # max_iter mulai dari 50 hingga 1049
    zoom = 1 + i * 0.05          # Faktor zoom meningkat secara linear
    center_x, center_y = -0.75, 0.0
    scale = 1.5 / zoom           # Luas area tampilan mengecil seiring zoom
    angle = i * 0.05             # Setiap gambar memiliki rotasi berbeda (dalam radian)

    # Hitung batas wilayah kompleks (tanpa rotasi, rotasi nanti diterapkan di fungsi)
    x_min = center_x - scale
    x_max = center_x + scale
    y_min = center_y - scale
    y_max = center_y + scale

    # Tentukan resolusi gambar (512 x 512)
    width, height = 512, 512

    # Generate fraktal Mandelbrot dengan rotasi menggunakan GPU (jika tersedia)
    fractal = generate_mandelbrot(x_min, x_max, y_min, y_max, width, height, max_iter, device, angle)

    # Normalisasi array ke rentang 0-255 untuk citra grayscale
    norm_fractal = (fractal.astype(np.float32) / fractal.max()) * 255
    norm_fractal = norm_fractal.astype(np.uint8)
    img = Image.fromarray(norm_fractal, mode='L')

    # Simpan gambar dengan nama berurutan, misalnya: mandelbrot_000.jpg, mandelbrot_001.jpg, dst.
    filename = os.path.join(output_folder, f"mandelbrot_{i:03d}.jpg")
    img.save(filename)
    print(f"Saved {filename}")