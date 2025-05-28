from PIL import Image

def resize_image(input_path, output_path, new_size):
    """
    Fungsi untuk meresize gambar.

    Parameter:
      - input_path: path file gambar input.
      - output_path: path file output yang akan disimpan.
      - new_size: tuple (width, height) ukuran baru gambar.
    """
    # Buka gambar
    image = Image.open(input_path)
    # Resize gambar dengan metode ANTIALIAS untuk kualitas yang lebih baik
    resized_image = image.resize(new_size, Image.Resampling.LANCZOS)
    # Simpan gambar yang sudah diresize
    resized_image.save(output_path)
    print(f"Gambar telah di-resize dan disimpan di {output_path}")

# Contoh penggunaan:
input_image = "D:\\14. Project Skripsi gw\\Batik (Style)\\batik-bali\\1.jpg"         # Ganti dengan path gambar asli
output_image = "Batik Bali 1_Resize.jpg"  # Nama file output
new_size = (512, 512)             # Ukuran baru (lebar x tinggi)
resize_image(input_image, output_image, new_size)