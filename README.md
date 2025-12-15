# 📘 Judul Proyek
*KLASIFIKASI CULTIVAR WINE MENGGUNAKAN MACHINE LEARNING PADA DATA WINE (UCI REPOSITORY)*

## 👤 Informasi
- **Nama:** RAHMAD RISKIAWAN H. SALEH
- **Repo:** (https://github.com/RahmadSaleh04/UAS_Wine_234311048_Rahmad-Riskiawan-H.-Saleh/tree/main)
- **Video:** [...]  

---

# 1. 🎯 Ringkasan Proyek
- Menyelesaikan permasalahan sesuai domain  
- Melakukan data preparation  
- Membangun 3 model: **Baseline**, **Advanced**, **Deep Learning**  
- Melakukan evaluasi dan menentukan model terbaik  

---

# 2. 📄 Problem & Goals
**Problem Statements:**  
- Bagaimana membangun model machine learning yang mampu mengklasifikasikan jenis wine berdasarkan kandungan kimia dengan tingkat akurasi yang tinggi?
- Algoritma machine learning apa yang memberikan performa terbaik pada dataset Wine untuk kasus klasifikasi multikelas?
- Apakah penggunaan model deep learning dapat meningkatkan performa klasifikasi dibandingkan dengan model baseline dan model machine learning konvensional?
- Fitur kimia apa saja yang paling berpengaruh dalam menentukan jenis wine?


**Goals:**  
- Membangun sistem klasifikasi wine menggunakan tiga pendekatan model, yaitu baseline model, advanced machine learning model, dan deep learning model.
- Mencapai akurasi klasifikasi minimal di atas 80% pada data uji.
- Membandingkan performa ketiga model berdasarkan metrik evaluasi yang relevan, seperti accuracy, precision, recall, dan F1-score.
- Menentukan model terbaik yang dapat digunakan untuk klasifikasi jenis wine secara efektif dan reproducible.


---
## 📁 Struktur Folder
```
project/
│
├── data/                   # Dataset (tidak di-commit, download manual)
│
├── notebooks/              # Jupyter notebooks
│   └── ML_Project.ipynb
│
├── src/                    # Source code
│   
├── models/                 # Saved models
│   ├── model_baseline.pkl
│   ├── model_rf.pkl
│   └── model_cnn.h5
│
├── images/                 # Visualizations
│   └── r
│
├── requirements.txt        # Dependencies
├── .gitignore
└── README.md
```
---

# 3. 📊 Dataset
- **Sumber:** https://archive.ics.uci.edu/dataset/109/wine
- **Jumlah Data:** 178 sampel
- **Tipe:** Tabular

### Fitur Utama
| Nama Fitur | Tipe Data | Deskripsi |
| :--- | :--- | :--- |
| **Alcohol** | Float | **Persentase kandungan alkohol** |
| **Malic Acid** | Float | **Kandungan asam Malik** |
| **Ash** | Float | **Residu hasil pembakaran** dalam gram |
| **Alcalinity of Ash** | Float | **Tingkat alkalinitas abu** |
| **Magnesium** | Float | **Kadar magnesium** (mg/L) |
| **Total Phenols** | Float | **Total senyawa fenol** |
| **Flavanoids** | Float | Senyawa **flavonoid** |
| **Nonflavanoid Phenols** | Float | Senyawa **fenol non-flavonoid** |
| **Proanthocyanins** | Float | Kandungan **proanthocyanins** |
| **Color Intensity** | Float | **Intensitas warna** |
| **Hue** | Float | **Value warna** (color hue) |
| **OD280/OD315** | Float | **Rasio absorbansi** pada panjang gelombang 280/315 |
| **Proline** | Float | Kandungan asam amino **Proline** |
| **Class (Label)** | Categorical | **Kelas wine** (0, 1, 2) berdasarkan *cultivar* |

---

# 4. 🔧 Data Preparation
- **Cleaning (missing/duplicate/outliers)**
  
**Handling Missing Values**
- Dataset Wine tidak memiliki missing values pada seluruh fitur.
- Oleh karena itu, tidak diperlukan proses imputasi data.
- eputusan ini diambil setelah melakukan pengecekan nilai null pada seluruh kolom.


**Removing Duplicates**
- Dataset diperiksa untuk kemungkinan data duplikat.
- Hasil pemeriksaan menunjukkan tidak terdapat data duplikat.
- Semua baris data bersifat unik.


**Handling Outliers**
- Dataset Wine memiliki beberapa nilai ekstrem pada fitur seperti Proline, Color Intensity, dan Alcohol.
- Outliers tidak dihapus, karena:
- Nilai ekstrem masih valid secara domain kimia.
- Model seperti Random Forest dan Neural Network relatif robust terhadap outliers.
- Data Type Conversion
- Seluruh fitur bertipe numerik (float dan int).
- Target kelas (class) dikonversi ke tipe kategorikal.

**Transformasi (encoding/scaling)**
- Karena dataset berbentuk tabular numerik, transformasi difokuskan pada scaling fitur.
**Encoding**
- Target class sudah berbentuk numerik (0,1,2).
- Tidak diperlukan One-Hot Encoding untuk model tree dan Logistic Regression.
- Untuk deep learning, target akan diubah ke one-hot encoding saat training.

**Scaling**
- Digunakan StandardScaler:
- Mean = 0
- Standard deviation = 1

**Alasan:**
- Logistic Regression dan MLP sensitif terhadap skala fitur.
- Membantu mempercepat konvergensi neural network.

- Splitting (train/val/test)  
- Dataset dibagi menjadi data latih dan data uji untuk mengevaluasi performa model.
- Strategi Pembagian Data
- Training set: 80% (142 sampel)
- Test set: 20% (36 sampel)
- Menggunakan stratified split untuk menjaga proporsi kelas.
- Random state = 42 untuk reproducibility.

---

# 5. 🤖 Modeling
- **Model 1 – Baseline:** Logistic Regression
- **Model 2 – Advanced ML:** Random Forest Classifier
- **Model 3 – Deep Learning:** Multilayer Perceptron (MLP)

---

# 6. 🧪 Evaluation
**Metrik:** Accuracy / F1 / MAE / MSE (pilih sesuai tugas)
Digunakan metrik standar untuk klasifikasi multikelas: **Accuracy, Precision, Recall,** dan **F1-Score**. **Confusion Matrix** juga digunakan untuk analisis kesalahan per-kelas.
 
### Hasil Singkat
| Model | Score | Catatan |
|-------|--------|---------|
| Baseline | 0.97 | Baik |
| Advanced | 1.0 | Sangat Baik |
| Deep Learning | 0.96 | Baik |

---

# 7. 🏁 Kesimpulan
* **Model Terbaik:** **Random Forest Classifier** memberikan performa superior, mencapai **Accuracy dan F1-Score 1.00** pada data uji.
* **Alasan:** Random Forest unggul dalam menangkap pola non-linear pada data tabular berukuran kecil, mengurangi *overfitting* melalui mekanisme *ensemble*, dan relatif *robust* terhadap *outliers*.

### Insight Penting

#### Insight dari Data
* Kualitas data sangat tinggi (tanpa missing/duplikat), memungkinkan fokus pada scaling dan splitting.
* Fitur **Flavanoids**, **Color Intensity**, dan **Proline** diperkirakan memiliki pengaruh terbesar dalam klasifikasi jenis wine (berdasarkan analisis Feature Importance dari Random Forest).

#### Insight dari Modeling
* Model *ensemble* (Random Forest) sangat efektif untuk dataset tabular berukuran kecil hingga menengah.
* Model *Deep Learning* (MLP), meskipun kuat, tidak selalu mengungguli model ML klasik pada dataset dengan jumlah sampel terbatas.


---

# 8. 🔮 Future Work
- [ ] Tambah data  
- [ ] Tuning model  
- [ ] Coba arsitektur DL lain  
- [ ] Deployment  

---

# 9. 🔁 Reproducibility
```
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2

# Deep Learning Framework (pilih salah satu)
tensorflow==2.14.0  # atau
torch==2.1.0        # PyTorch

# Additional libraries (sesuaikan)
xgboost==1.7.6
lightgbm==4.0.0
opencv-python==4.8.0  # untuk computer vision
nltk==3.8.1           # untuk NLP
transformers==4.30.0  # untuk BERT, dll

```
