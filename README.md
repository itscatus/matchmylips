# MatchMyLips

**MatchMyLips** adalah sebuah aplikasi berbasis web yang dikembangkan menggunakan Streamlit dengan tujuan untuk membantu pengguna memilih rona pewarna bibir yang sesuai dengan warna kulit mereka berdasarkan analisis Seasonal Personal Color menggunakan model Convolutional Neural Network (CNN). Aplikasi ini juga menyediakan fitur try-on untuk mencoba rona pewarna bibir secara real-time via foto.

---

## Fitur Utama
1. **Rekomendasi Shade Lipstik**:
   - Deteksi Seasonal Personal Color berdasarkan selfie yang diunggah.
   - Rekomendasi shade lipstik yang sesuai dengan warna kulit.

2. **Virtual Try-On**:
   - Mengunggah gambar shade lipstik untuk mendeteksi kategori Seasonal Personal Color.
   - Mencoba shade lipstik pada selfie yang dianalisis.
   - Virtual try-on menggunakan webcam (dapat diakses setelah pengguna mengunggah gambar shade lipstik).

3. **Pengelolaan Gambar Shade Lipstik**:
   - Pengguna dapat mengunggah gambar shade lipstik untuk dianalisis.

---

## Teknologi yang Digunakan
- **Frontend**:
  - Framework: Streamlit
  - Library: OpenCV (untuk pengolahan gambar)

- **Backend**:
  - Framework: PyTorch (untuk model CNN)
  - Python

---

## Cara Kerja Aplikasi
1. **Deteksi Seasonal Personal Color**:
   - Pengguna mengunggah foto citra wajah ke aplikasi.
   - Aplikasi menganalisis gambar menggunakan model CNN untuk mendeteksi Seasonal Personal Color.
   
2. **Rekomendasi Shade Lipstik**:
   - Setelah analisis, aplikasi merekomendasikan rona pewarna bibir yang sesuai dengan kategori warna personal (Spring, Summer, Autumn, atau Winter).

3. **Shade Check**
   - Pengguna dapat mengunggah citra kumpulan rona pewarna bibir untuk mendeteksi kategori Seasonal Personal Color.
   - Mendapatkan kode Hex dari warna pada rona yang dipilih.


4. **Virtual Try-On**:
   - Rona pewarna bibir dapat dicoba secara langsung pada citra wajah.
   - Menggunakan perhitungan alpha untuk transparency

---

## Cara Instalasi dan Menjalankan Aplikasi

1. **Clone Repository**:
   ```bash
   git clone https://github.com/itscatus/websheesh.git
   cd matchmylips
   ```

2. **Instalasi Dependensi**:
   Pastikan Anda memiliki Python dan pip terinstal. Jalankan:
   ```bash
   pip install -r requirements.txt
   ```

3. **Menjalankan Aplikasi**:
   ```bash
   streamlit run Home.py
   ```

4. **Akses Aplikasi**:
   - Buka browser dan akses: `http://localhost:8501`

---

## Struktur Proyek
```
matchmylips/
├── app.py               # Main script untuk menjalankan aplikasi
├── model/
│   └── personal_color_model.h5  # File model CNN
├── data/
│   ├── example_selfie.jpg   # Contoh gambar selfie
│   └── lip_shades/          # Folder gambar shade lipstik
├── utils/
│   ├── preprocessing.py     # Skrip untuk preprocessing gambar
│   └── analysis.py          # Skrip untuk analisis warna personal
├── requirements.txt     # Daftar dependensi Python
└── README.md            # Dokumentasi proyek
```

---

## Pengembang
- Nama: [Nama Anda]
- Universitas: [Universitas Anda]
- Kontak: [Email atau LinkedIn Anda]

---

## Lisensi
Aplikasi ini dirilis di bawah lisensi MIT. Silakan merujuk ke file LICENSE untuk detail lebih lanjut.

