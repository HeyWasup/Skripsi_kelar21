# Skripsi_kelar21
Skripsi cepat kelar ya Allah.......


# Neural Style Transfer Batik–Fraktal

Repository ini berisi skrip untuk melakukan **iterative Gatys-style NST** dengan VGG19 yang telah di–fine-tune untuk domain batik vs fraktal.

---

## 📦 Dependencies

Semua dependensi tercantum di `requirements.txt`. Contoh isinya:
torch>=1.13.0
torchvision>=0.14.0
pillow>=9.0.0
matplotlib>=3.5.0


> `tkinter` biasanya sudah tersedia di instalasi Python standar.

---

## 🛠️ Setup Virtual Environment

1. **Clone** repository ini  
   ```bash
   git clone https://github.com/username/proyek-nst.git
   cd proyek-nst

## Buat dan aktifkan virtual environment

> linux/macOs
python3 -m venv .venv
source .venv/bin/activate
> Windows
python -m venv .venv
.venv\Scripts\activate

## Upgrade pip dan install semua paket
pip install --upgrade pip
pip install -r requirements.txt

## Cara Menjalankan Script
Setelah virtualenv aktif, jalankan:
python gatys_nst_finetuned.py
