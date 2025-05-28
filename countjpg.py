import os

def count_jpg_files(folders):
    if len(folders) > 3:
        print("Maksimal hanya dapat memasukkan 3 folder.")
        return
    
    for folder in folders:
        if os.path.exists(folder) and os.path.isdir(folder):
            jpg_files = [f for f in os.listdir(folder) if f.lower().endswith('.jpg')]
            print(f"Folder '{folder}' memiliki {len(jpg_files)} file JPG.")
        else:
            print(f"Folder '{folder}' tidak ditemukan atau bukan direktori yang valid.")

if __name__ == "__main__":
    folders = ["D:/14. Project Skripsi gw/batik semua"]  # Ganti dengan path folder yang sesuai
    count_jpg_files(folders)