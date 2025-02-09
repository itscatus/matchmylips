<img src="assets/logo.png" width="100" />

# MatchMyLips

**MatchMyLips** is a web-based application developed using Streamlit with the aim of helping users choose the right lip color shade based on Seasonal Personal Color analysis using the Convolutional Neural Network (CNN) model. This application also provides a try-on feature to try lip color shades via uploaded facial images.

---

## Key Features
1. **Seasonal Personal Color Detection from User's Facial Image**:
- Users upload a photo of their facial image to the application.
- The application analyzes the image using a CNN model prepared to detect Seasonal Personal Color.
- The application provides recommendations for lip color shades that match the type of Seasonal Personal Color detected.

![](img/spca_check.png)

2. **Seasonal Personal Color Detection from Lip Color Shades**:
- Users upload a photo of their lip color shade image to the application.
- The application analyzes the Seasonal Personal Color category of the clicked color hue.
- The results are given in the form of a Hex code of the color along with its Seasonal Personal Color type.

![](img/shade-check.png)

3. **Virtual Try-on**:
- The user uploads a photo of a facial image to the application.
- The user selects the Hex color code to be applied to the lips
- The application will process and apply the selected Hex color code to the lip image using masking with the alpha blending technique.

![](img/try-on.png)

---

## Technologies Used
- **Frontend**:
- Framework: Streamlit
- Library: OpenCV (for image processing)

- **Backend**:
- Framework: PyTorch (for CNN model)
- Python

---

## How to Install and Run the Application

1. **Clone Repository**:
```bash
git clone https://github.com/itscatus/websheesh.git
cd matchmylips
```

2. **Installation Dependencies**:
Make sure you have Python and pip installed. Then run:
```bash
pip install -r requirements.txt
```

3. **Running the Application**:
```bash
streamlit run Home.py
```

4. **Application Access**:
- Open a browser and access: `http://localhost:8501`

---

## Project Structure
```
matchmylips/
├── .streamlit/
│ └── config.toml   # Script to set the streamlit display
├── Home.py   # Main script to run the application and display the homepage
├── assets/
│ ├── colors.csv              # Lip color hue dataset file
│ ├── cosmetics.png           # Image file for the Shade Check page icon
│ ├── house-door.png          # Image file for Home page icon
│ ├── lips.png                # Image file for Try-on Lippies page icon
│ ├── logo.png                # Image file for application logo
│ └── person-bounding-box.png # Image file for Personal Color Analysis page icon
├── facer/   # Library model for face detection and parsing
├── pages/
│ ├── Personal_Color_Analysis.py   # Script for personal color analysis page
│ ├── Shade_Check.py               # Script for lip color check page
│ └── Try-on_Lippies.py            # Script for color try-on page lips
├── utils/
│ ├── functions.py       # Script to check whether a face is detected or not
│ └── translations.py    # Indonesian and English translation dictionary script
├── requirements.txt   # List of Python dependencies
├── best_mobilenetv2_model.pth   # Mobilenetv2 model file used for personal color analysis prediction from facial skin images
└── README.md   # Project documentation
```

---

## Developer
- Name: Felicia Natania Lingga
- University: Universitas Padjadjaran
- Contact: felicia21001@mail.unpad.ac.id

---

## License
This application is released under the MIT license. Please refer to the LICENSE file for more details.

---

<img src="assets/logo.png" width="100" />

# MatchMyLips

**MatchMyLips** adalah sebuah aplikasi berbasis web yang dikembangkan menggunakan Streamlit dengan tujuan untuk membantu pengguna memilih rona pewarna bibir yang sesuai berdasarkan analisis Seasonal Personal Color menggunakan model Convolutional Neural Network (CNN). Aplikasi ini juga menyediakan fitur try-on untuk mencoba rona pewarna bibir via citra wajah yang diunggah.

---

## Fitur Utama
1. **Deteksi Seasonal Personal Color dari Citra Wajah Pengguna**:
   - Pengguna mengunggah foto citra wajah ke aplikasi.
   - Aplikasi menganalisis citra menggunakan model CNN yang disiapkan untuk mendeteksi Seasonal Personal Color.
   - Aplikasi memberikan rekomendasi rona pewarna bibir yang sesuai dengan tipe Seasonal Personal Color yang dideteksi.

![](img/spca_check.png)

2. **Deteksi Seasonal Personal Color dari Rona Pewarna Bibir**:
   - Pengguna mengunggah foto citra rona pewarna bibir ke aplikasi.
   - Aplikasi menganalisis kategori Seasonal Personal Color dari rona warna yang di-klik.
   - Hasil yang diberikan berupa kode Hex dari warna beserta tipe Seasonal Personal Colornya.

![](img/shade-check.png)

3. **Virtual Try-on**:
   - Pengguna mengunggah foto citra wajah ke aplikasi.
   - Pengguna memilih kode warna Hex yang ingin diterapkan pada bibir
   - Aplikasi akan memproses dan menerapkan kode warna Hex yang dipilih pada citra bibir menggunakan masking dengan teknik alpha blending.

![](img/try-on.png)

---

## Teknologi yang Digunakan
- **Frontend**:
  - Framework: Streamlit
  - Library: OpenCV (untuk pengolahan gambar)

- **Backend**:
  - Framework: PyTorch (untuk model CNN)
  - Python

---

## Cara Instalasi dan Menjalankan Aplikasi

1. **Clone Repository**:
   ```bash
   git clone https://github.com/itscatus/websheesh.git
   cd matchmylips
   ```

2. **Instalasi Dependensi**:
   Pastikan Anda memiliki Python dan pip terinstal. Kemudian jalankan:
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
├── .streamlit/
│   └── config.toml   # Script untuk mengatur tampilan streamlit
├── Home.py   # Main script untuk menjalankan aplikasi dan menampilkan homepage
├── assets/
│   ├── colors.csv               # File dataset rona pewarna bibir
│   ├── cosmetics.png            # File image untuk page icon halaman Shade Check
│   ├── house-door.png           # File image untuk page icon halaman Home
│   ├── lips.png                 # File image untuk page icon halaman Try-on Lippies
│   ├── logo.png                 # File image untuk logo aplikasi
│   └── person-bounding-box.png  # File image untuk page icon halaman Personal Color Analysis
├── facer/   # Library model untuk deteksi dan parsing wajah
├── pages/
│   ├── Personal_Color_Analysis.py   # Script untuk halaman personal color analysis 
│   ├── Shade_Check.py               # Script untuk halaman cek pewarna bibir
│   └── Try-on_Lippies.py            # Script untuk halaman try-on pewarna bibir
├── utils/
│   ├── functions.py      # Script untuk mengecek apakah wajah terdeteksi atau tidak
│   └── translations.py   # Script dictionary terjemahan bahasa Indonesia dan bahasa Inggris
├── requirements.txt   # Daftar dependensi Python
├── best_mobilenetv2_model.pth    # File model mobilenetv2 yang digunakan untuk prediksi personal color analysis dari citra kulit wajah  
└── README.md   # Dokumentasi proyek
```

---

## Pengembang
- Nama: Felicia Natania Lingga
- Universitas: Universitas Padjadjaran
- Kontak: felicia21001@mail.unpad.ac.id

---

## Lisensi
Aplikasi ini dirilis di bawah lisensi MIT. Silakan merujuk ke file LICENSE untuk detail lebih lanjut.

