import os
import random
import shutil

# Tentukan folder sumber dan folder tujuan
source_folder = "D:/14. Project Skripsi gw/batik gw"
folder_a = 'batik_train'  # untuk 80% file
folder_b = 'batik_val'  # untuk 20% file

# Buat folder tujuan jika belum ada
os.makedirs(folder_a, exist_ok=True)
os.makedirs(folder_b, exist_ok=True)

# Dapatkan daftar semua file JPG di folder sumber
files = [f for f in os.listdir(source_folder) if f.lower().endswith('.jpg')]

# Acak daftar file agar pembagian bersifat acak
random.shuffle(files)

# Hitung jumlah total file dan tentukan batas pembagian 80/20
total_files = len(files)
split_index = int(total_files * 0.8)

# Bagi file menjadi dua list: 80% untuk folder_a dan 20% untuk folder_b
files_a = files[:split_index]
files_b = files[split_index:]

# Pindahkan file ke folder_a
for file in files_a:
    src = os.path.join(source_folder, file)
    dst = os.path.join(folder_a, file)
    shutil.move(src, dst)

# Pindahkan file ke folder_b
for file in files_b:
    src = os.path.join(source_folder, file)
    dst = os.path.join(folder_b, file)
    shutil.move(src, dst)

print(f"Berhasil memindahkan {len(files_a)} file ke folder '{folder_a}' dan {len(files_b)} file ke folder '{folder_b}'.")
