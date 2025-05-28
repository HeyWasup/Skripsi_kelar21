import torch
import numpy as np
from PIL import Image
import os

def generate_mandelbrot(x_min, x_max, y_min, y_max, width, height, max_iter, device, angle=0):
    """
    Menghasilkan array 2D berisi nilai iterasi Mandelbrot untuk setiap piksel menggunakan PyTorch,
    dengan rotasi tambahan terhadap koordinat.
    """
    # Buat koordinat linier
    x = torch.linspace(x_min, x_max, width, device=device)
    y = torch.linspace(y_min, y_max, height, device=device)
    Y, X = torch.meshgrid(y, x, indexing='ij')
    
    # Rotasi koordinat
    cos_a = torch.cos(torch.tensor(angle, device=device))
    sin_a = torch.sin(torch.tensor(angle, device=device))
    Xr = X * cos_a - Y * sin_a
    Yr = X * sin_a + Y * cos_a
    
    # Komplek C, Z, dan mask iterasi
    C = Xr + 1j * Yr
    Z = torch.zeros_like(C, device=device)
    M = torch.full(C.shape, max_iter, device=device, dtype=torch.int32)

    for i in range(max_iter):
        mask = torch.abs(Z) <= 2
        Z[mask] = Z[mask]**2 + C[mask]
        diverged = mask & (torch.abs(Z) > 2)
        M[diverged] = i
    return M.cpu().numpy()

# Set device PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Menggunakan device:", device)

# Folder output
output_folder = "MandelbrotOutputs(add Rotation,512res,notext&thick)"
os.makedirs(output_folder, exist_ok=True)

# Ketebalan garis (berapa kali dilasi; 0 = no dilation)
thickness = 1

# Loop generate 1000 fraktal
for i in range(1000):
    max_iter   = 50 + i
    zoom       = 1 + i * 0.05
    center_x   = -0.75
    center_y   = 0.0
    scale      = 1.5 / zoom
    angle      = i * 0.05

    # Batas koordinat
    x_min, x_max = center_x - scale, center_x + scale
    y_min, y_max = center_y - scale, center_y + scale
    width, height = 512, 512

    # Generate mandelbrot
    fractal = generate_mandelbrot(x_min, x_max, y_min, y_max,
                                  width, height, max_iter, device, angle)

    # Normalisasi ke 0–255
    norm = (fractal.astype(np.float32) / fractal.max()) * 255
    norm = norm.astype(np.uint8)

    # ——— Penebalan garis fraktal ———
    # 1. Deteksi edge (perubahan iterasi antar-piksel)
    edge = np.zeros_like(fractal, dtype=bool)
    edge[1:, :]   |= fractal[1:, :]   != fractal[:-1, :]
    edge[:-1, :]  |= fractal[:-1, :]  != fractal[1:, :]
    edge[:, 1:]   |= fractal[:, 1:]   != fractal[:, :-1]
    edge[:, :-1]  |= fractal[:, :-1]  != fractal[:, 1:]

    # 2. Dilasi sederhana untuk menambah ketebalan
    edge_d = edge.copy()
    for _ in range(thickness):
        edge_d = (
            edge_d |
            np.roll(edge_d,  1, axis=0) |
            np.roll(edge_d, -1, axis=0) |
            np.roll(edge_d,  1, axis=1) |
            np.roll(edge_d, -1, axis=1)
        )

    # 3. Overlay: garis tebal jadi hitam (0)
    norm[edge_d] = 0

    # Simpan gambar
    img = Image.fromarray(norm, mode='L')
    filename = os.path.join(output_folder, f"mandelbrot_{i:03d}.jpg")
    img.save(filename)
    print(f"Saved {filename}")