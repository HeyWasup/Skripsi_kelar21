import os
import hashlib
from PIL import Image, UnidentifiedImageError
from collections import defaultdict

def hash_image(file_path):
    """Menghitung hash MD5 untuk file gambar."""
    with open(file_path, 'rb') as f:
        data = f.read()
        return hashlib.md5(data).hexdigest()

def validate_images(root_dir):
    """
    Memeriksa setiap file gambar di root_dir:
      - Mencoba membuka gambar, dan mencetak warning jika gagal.
      - Menghitung hash untuk mendeteksi duplikasi.
      - Menghitung jumlah gambar per kelas (berdasarkan nama subfolder).
    """
    image_hashes = {}
    class_counts = defaultdict(int)
    invalid_images = []
    duplicate_images = []

    for subdir, dirs, files in os.walk(root_dir):
        # Gunakan nama folder terakhir sebagai label kelas, misalnya: batik, fractal, dll.
        label = os.path.basename(subdir)
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                file_path = os.path.join(subdir, file)
                try:
                    # Coba buka gambar
                    with Image.open(file_path) as img:
                        img.verify()  # Memastikan file tidak corrupt
                    # Jika berhasil, hitung hash
                    img_hash = hash_image(file_path)
                    if img_hash in image_hashes:
                        duplicate_images.append((file_path, image_hashes[img_hash]))
                    else:
                        image_hashes[img_hash] = file_path
                        class_counts[label] += 1
                except (UnidentifiedImageError, Exception) as e:
                    print(f"Warning: Gagal membuka {file_path}. Error: {e}")
                    invalid_images.append(file_path)
    return class_counts, invalid_images, duplicate_images

if __name__ == '__main__':
    # Ganti dengan path dataset Anda (misalnya, dataset gabungan dengan subfolder sesuai kelas)
    dataset_path = "D:/14. Project Skripsi gw/batik semua"
    
    class_counts, invalid_images, duplicate_images = validate_images(dataset_path)
    
    print("\nJumlah gambar per kelas:")
    for label, count in class_counts.items():
        print(f"{label}: {count}")
    
    if invalid_images:
        print("\nGambar yang tidak valid:")
        for path in invalid_images:
            print(path)
    else:
        print("\nSemua gambar valid.")
    
    if duplicate_images:
        print("\nDuplikasi gambar ditemukan:")
        for dup in duplicate_images:
            print(f"{dup[0]} adalah duplikat dari {dup[1]}")
    else:
        print("\nTidak ada gambar duplikat.")