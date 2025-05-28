import os
import shutil

def combine_images(input_parent_folder, output_folder):
    """
    Menggabungkan semua gambar dari subfolder di input_parent_folder ke output_folder.
    Setiap gambar akan di-copy dengan nama baru: <nama_file_asli>_(<nama_subfolder>).<ekstensi>
    """
    # Buat folder output jika belum ada
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Folder output '{output_folder}' dibuat.")
    else:
        print(f"Folder output '{output_folder}' sudah ada.")
    
    # Iterasi setiap subfolder di dalam folder induk
    subfolders = [f for f in os.listdir(input_parent_folder) if os.path.isdir(os.path.join(input_parent_folder, f))]
    if not subfolders:
        print("Tidak ada subfolder di dalam folder induk.")
    
    for subfolder in subfolders:
        subfolder_path = os.path.join(input_parent_folder, subfolder)
        print(f"Memproses subfolder: {subfolder}")
        files = os.listdir(subfolder_path)
        print(f"Jumlah file di {subfolder}: {len(files)}")
        
        # Iterasi setiap file di subfolder
        for filename in files:
            # Pastikan file merupakan gambar (ekstensi .jpg, .jpeg, .png)
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                name, ext = os.path.splitext(filename)
                new_filename = f"{name}_({subfolder}){ext}"
                src_path = os.path.join(subfolder_path, filename)
                dest_path = os.path.join(output_folder, new_filename)
                try:
                    shutil.copy2(src_path, dest_path)
                    print(f"Copied: {src_path} -> {dest_path}")
                except Exception as e:
                    print(f"Error menyalin {src_path}: {e}")
            else:
                print(f"Dilewati (bukan gambar): {filename}")

if __name__ == '__main__':
    # Ganti dengan path folder dataset induk Anda
    input_parent_folder = "D:/14. Project Skripsi gw/awal"
    # Ganti dengan path folder output yang diinginkan
    output_folder = "D:/14. Project Skripsi gw/akhir"
    
    combine_images(input_parent_folder, output_folder)