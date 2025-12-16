## INFORMASI PROYEK

**Judul Proyek:**  
KLASIFIKASI CULTIVAR WINE MENGGUNAKAN MACHINE LEARNING PADA DATA WINE (UCI REPOSITORY)

* **Nama Mahasiswa:** RAHMAD RISKIAWAN H. SALEH
* **NIM:** 234311048
* **Program Studi:** Rekayasa Perangkat Lunak
* **Mata Kuliah:** Data Science
* **Dosen Pengampu:** Gus Nanang Syaifuddiin, S.Kom., M.Kom.
* **Tahun Akademik:** 2025/Semester 5
* **Link GitHub Repository:** [URL Repository]
* **Link Video Pembahasan:** [URL Repository]

---

## 1. LEARNING OUTCOMES
Pada proyek ini, mahasiswa diharapkan dapat:
* Memahami perumusan *problem statement* dalam masalah klasifikasi.
* Melakukan analisis dan eksplorasi data (*EDA*) secara komprehensif.
* Melakukan *preprocessing dataset* dan penyesuaian fitur sesuai karakteristik data.
* Mengembangkan tiga model *machine learning* yang terdiri dari:
    * Model *baseline*
    * Model *advanced*
    * Model *deep learning* (MLP)
* Menggunakan metrik evaluasi yang relevan untuk tugas klasifikasi multikelas.
* Melaporkan hasil eksperimen secara ilmiah dan sistematis.
* Mengunggah seluruh kode proyek ke GitHub (*public repository*).
* Menerapkan prinsip *software engineering* dalam struktur proyek ML.

---

## 2. PROJECT OVERVIEW

### 2.1 Latar Belakang
Industri *wine* secara tradisional menilai kualitas melalui pengujian laboratorium dan evaluasi sensorik oleh pakar. Metode manual ini bersifat subjektif dan membutuhkan waktu. Dengan kemajuan *machine learning*, analisis kimia *wine* dapat digunakan untuk memprediksi kelas *wine* secara otomatis dan akurat berdasarkan kandungan kimianya.

Dataset **Wine dari UCI Machine Learning Repository** berisi 178 sampel *wine* dari tiga kelas (cultivar) berbeda, dengan masing-masing 13 karakteristik kimia. Dataset ini menjadi *benchmark* karena komposisi fiturnya yang jelas dan sifat multikelasnya.

Pemanfaatan algoritma *machine learning* untuk prediksi kelas *wine* secara otomatis bermanfaat untuk:
* Mendukung produsen *wine* dalam kontrol kualitas produk.
* Mengurangi biaya evaluasi manual.
* Menjadi contoh penerapan *predictive analytics* berbasis data kimia.

**Contoh referensi (berformat APA/IEEE):**
> Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.


## 3. BUSINESS UNDERSTANDING / PROBLEM UNDERSTANDING
### 3.1 Problem Statements

Permasalahan yang ingin diselesaikan dalam proyek ini adalah:
1.  Bagaimana membangun model *machine learning* yang mampu mengklasifikasikan jenis *wine* berdasarkan kandungan kimia dengan tingkat akurasi yang tinggi?
2.  Algoritma *machine learning* apa yang memberikan performa terbaik pada *dataset Wine* untuk kasus klasifikasi multikelas?
3.  Apakah penggunaan model *deep learning* dapat meningkatkan performa klasifikasi dibandingkan dengan model *baseline* dan model *machine learning* konvensional?
4.  Fitur kimia apa saja yang paling berpengaruh dalam menentukan jenis *wine*?


### 3.2 Goals

Tujuan proyek:
* Membangun sistem klasifikasi *wine* menggunakan tiga pendekatan model: *baseline model*, *advanced machine learning model*, dan *deep learning model*.
* Mencapai akurasi klasifikasi minimal di atas **80%** pada data uji.
* Membandingkan performa ketiga model berdasarkan metrik evaluasi yang relevan, seperti *accuracy, precision, recall*, dan *F1-score*.
* Menentukan model terbaik yang dapat digunakan untuk klasifikasi jenis *wine* secara efektif dan *reproducible*.


### 3.3 Solution Approach

| Model | Nama Model | Alasan Pemilihan |
| :--- | :--- | :--- |
| **Model 1** | **Logistic Regression** | Dipilih sebagai *baseline* karena merupakan algoritma klasifikasi linear yang sederhana, mudah diinterpretasikan, dan cocok untuk data tabular dengan fitur yang tidak terlalu besar. |
| **Model 2** | **Random Forest Classifier** | Dipilih sebagai model lanjutan karena mampu menangani hubungan non-linear, mengurangi *overfitting* melalui mekanisme *ensemble*, dan menyediakan informasi *feature importance*. |
| **Model 3** | **Multilayer Perceptron (MLP)** | Dipilih untuk memenuhi *requirement deep learning*. Model ini terdiri dari minimal dua *hidden layer* dan mampu mempelajari representasi fitur yang lebih kompleks, diharapkan dapat meningkatkan performa. |

**Minimum Requirements untuk Deep Learning:**
- ✅ Model harus training minimal 10 epochs
- ✅ Harus ada plot loss dan accuracy/metric per epoch
- ✅ Harus ada hasil prediksi pada test set
- ✅ Training time dicatat (untuk dokumentasi)

**Tidak Diperbolehkan:**
- ❌ Copy-paste kode tanpa pemahaman
- ❌ Model tidak di-train (hanya define arsitektur)
- ❌ Tidak ada evaluasi pada test set


---

## 4. DATA UNDERSTANDING
### 4.1 Informasi Dataset
**Sumber Dataset:**  
[Sebutkan sumber: Kaggle, UCI ML Repository, atau sumber lain dengan URL]

**Deskripsi Dataset:**
- Jumlah baris (rows): 178
- Jumlah kolom (columns/features): 13 fitur + 1 target (class)
- Tipe data: Tabular
- Ukuran dataset: 0.02MB
- Format file: CSV / array numerik 
### 4.2 Deskripsi Fitur
Jelaskan setiap fitur/kolom yang ada dalam dataset.
| Nama Fitur | Tipe | Deskripsi |
| :--- | :--- | :--- |
| Alcohol | Float | Persentase kandungan alkohol |
| Malic Acid | Float | Kandungan asam Malik |
| Ash | Float | Residu hasil pembakaran dalam gram |
| Alcalinity of Ash | Float | Tingkat alkalinitas abu |
| Magnesium | Float | Kadar magnesium (mg/L) |
| Total Phenols | Float | Total senyawa fenol |
| Flavanoids | Float | Senyawa flavonoid |
| Nonflavanoid Phenols | Float | Senyawa fenol non-flavonoid |
| Proanthocyanins | Float | Kandungan proanthocyanins |
| Color Intensity | Float | Intensitas warna |
| Hue | Float | *Value* warna (*color hue*) |
| OD280/OD315 | Float | Rasio absorbansi pada panjang gelombang 280/315 |
| Proline | Float | Kandungan asam amino Proline |
| **Class (Label)** | Categorical | Kelas *wine* (0, 1, 2) berdasarkan cultivar |


### 4.3 Kondisi Data

Jelaskan kondisi dan permasalahan data:

* **Missing Values:** Tidak ada (*0%*)
* **Duplicate Data:** Tidak ditemukan.
* **Outliers:** Ada indikasi *outliers* pada fitur **proline**, **color\_intensity**, dan **magnesium**, namun dianggap wajar sebagai variasi alami.
* **Imbalanced Data:** Distribusi kelas relatif seimbang.
* **Data Quality Issues:** Tidak ada masalah signifikan.

### 4.4 Exploratory Data Analysis (EDA) - (**OPSIONAL**)

**Requirement:** Minimal 3 visualisasi yang bermakna dan insight-nya.
**Contoh jenis visualisasi yang dapat digunakan:**
- Histogram (distribusi data)
- Boxplot (deteksi outliers)
- Heatmap korelasi (hubungan antar fitur)
- Bar plot (distribusi kategori)
- Scatter plot (hubungan 2 variabel)
- Wordcloud (untuk text data)
- Sample images (untuk image data)
- Time series plot (untuk temporal data)
- Confusion matrix heatmap
- Class distribution plot


#### Visualisasi 1: [Judul Visualisasi]
[Insert gambar/plot]

**Insight:**  
* **Insight:** Dataset memiliki tiga kelas dengan jumlah sampel yang relatif seimbang, menunjukkan tidak adanya *class imbalance*.

#### Visualisasi 2: [Judul Visualisasi]

[Insert gambar/plot]

**Insight:**  
* **Insight:** Sebagian besar data uji berhasil diklasifikasikan dengan benar, ditandai dengan nilai diagonal yang dominan (akurasi tinggi). Model memiliki kemampuan generalisasi yang baik.

#### Visualisasi 3: [Judul Visualisasi]

[Insert gambar/plot]

**Insight:**  
* **Insight:** Terdapat pemisahan pola antar kelas *wine* berdasarkan kombinasi fitur **Alcohol** dan **Color Intensity**. Setiap kelas cenderung membentuk klaster tersendiri, menunjukkan kontribusi signifikan kedua fitur dalam klasifikasi.



---

## 5. DATA PREPARATION

### 5.1 Data Cleaning
**Aktivitas:**
* **Handling Missing Values:** Tidak ada *missing values*, sehingga tidak diperlukan imputasi.
* **Removing Duplicates:** Tidak terdapat data duplikat, semua baris unik.
* **Handling Outliers:** *Outliers* pada **Proline**, **Color Intensity**, dan **Alcohol** **tidak dihapus** karena masih valid secara domain kimia dan model yang digunakan (*Random Forest*, *Neural Network*) relatif *robust*.
* **Data Type Conversion:** Fitur numerik dipertahankan; *target class* dikonversi ke tipe kategorikal.



### 5.2 Feature Engineering
**Aktivitas:**
* **Feature Selection:** Semua 13 fitur kimia dipertahankan karena tidak ada fitur yang redundan ekstrem dan semuanya berkontribusi terhadap klasifikasi.
* **Dimensionality Reduction:** **PCA** hanya digunakan untuk visualisasi EDA, bukan untuk *training* model.


### 5.3 Data Transformation

**Untuk Data Tabular:**
* **Encoding:** *Target class* sudah numerik (0, 1, 2). Untuk *deep learning*, target akan diubah ke *one-hot encoding* saat *training*.
* **Scaling:** Digunakan **StandardScaler** ($\text{Mean} = 0$, $\text{Standard Deviation} = 1$).
    * **Alasan:** Logistic Regression dan MLP sensitif terhadap skala fitur, dan membantu mempercepat konvergensi *neural network*.


### 5.4 Data Splitting

**Strategi pembagian data:**
* **Strategi:** *Training set* **80%** (142 sampel), *Test set* **20%** (36 sampel).
* Menggunakan **stratified split** untuk menjaga proporsi kelas.


### 5.5 Data Balancing (jika diperlukan)
* **Tidak diperlukan** teknik *balancing* (seperti SMOTE atau *class weights*) karena distribusi kelas dianggap tidak *imbalanced* (Class 0: 59, Class 1: 71, Class 2: 48).

### 5.6 Ringkasan Data Preparation

| Langkah | Penjelasan | Apa yang Dilakukan | Mengapa Penting | Bagaimana Implementasi |
| :--- | :--- | :--- | :--- | :--- |
| **Pembersihan Data (*Data Cleaning*)** | Memastikan kualitas dan konsistensi data sebelum pemrosesan lebih lanjut. |  Memverifikasi tidak ada *missing values* (`NaN`).  Memastikan tidak ada data duplikat.  Mengabaikan penghapusan *outliers* karena model yang digunakan *robust*. | Data yang bersih menghasilkan model yang lebih stabil, akurat, dan memiliki kemampuan generalisasi yang baik. | Menggunakan fungsi dari **Pandas** (misalnya `df.isnull().sum()`, `df.drop_duplicates()`) dan analisis statistik. |
| **Scaling Fitur (*Feature Scaling*)** | Menyamakan rentang nilai pada semua fitur agar tidak ada fitur yang mendominasi proses *training*. | Mengaplikasikan **StandardScaler** (Z-score normalization) pada semua 13 fitur kimia. | Model berbasis jarak dan gradien (**Logistic Regression, MLP**) sangat sensitif terhadap skala fitur. *Scaling* mempercepat konvergensi *neural network*. | Menggunakan `StandardScaler` dari **Scikit-learn** (`sklearn.preprocessing`). |
| **Pembagian Data (*Data Splitting*)** | Memisahkan data menjadi set pelatihan dan set pengujian untuk evaluasi performa model yang tidak bias (*unseen data*). | Membagi data menjadi 80% untuk *Training Set* dan 20% untuk *Test Set*. Menggunakan metode **stratified** untuk menjaga proporsi kelas. | Menilai kemampuan generalisasi model secara akurat. Pembagian *stratified* menjaga distribusi kelas tetap seimbang di setiap set data. | Menggunakan fungsi `train_test_split` dari **Scikit-learn** (`sklearn.model_selection`). |

---

## 6. MODELING
### 6.1 Model 1 — Baseline Model
#### 6.1.1 Deskripsi Model

**Nama Model:** Logistic Regression
**Teori Singkat:**  
 Logistic Regression merupakan model klasifikasi linear yang memodelkan probabilitas suatu kelas menggunakan fungsi logistik (sigmoid/softmax). Untuk kasus multikelas, digunakan pendekatan one-vs-rest atau multinomial.
 
**Alasan Pemilihan:**  
Model ini dipilih sebagai baseline karena:
- Sederhana dan mudah diinterpretasikan
- Memberikan gambaran performa awal dataset
- Sering digunakan sebagai pembanding awal dalam studi klasifikasi


#### 6.1.2 Hyperparameter
**Parameter yang digunakan:**
```
max_iter: 500
solver: lbfgs
multi_class: auto
```

#### 6.1.3 Implementasi (Ringkas)
```python
import joblib
import os


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


model_lr = LogisticRegression(max_iter=500)
model_lr.fit(X_train, y_train)


y_pred_lr = model_lr.predict(X_test)


print("Accuracy Logistic Regression:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))


# Create the 'models' directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')


joblib.dump(model_lr, 'models/logistic_regression_wine.pkl')
loaded_lr = joblib.load('models/logistic_regression_wine.pkl')
```

#### 6.1.4 Hasil Awal

**[Tuliskan hasil evaluasi awal, akan dijelaskan detail di Section 7]**

---

### 6.2 Model 2 — ML / Advanced Model
#### 6.2.1 Deskripsi Model

**Nama Model:**  Random Forest 
**Teori Singkat:**  
 Random Forest adalah metode ensemble berbasis decision tree yang membangun banyak pohon keputusan dan menggabungkan hasil prediksi melalui voting mayoritas. Model ini mampu menangkap hubungan non-linear dan mengurangi overfitting.

**Alasan Pemilihan:**  
- Sangat efektif untuk data tabular Tidak sensitif terhadap outliers
- Menyediakan feature importance


**Keunggulan:**
- Akurasi tinggi
- Robust terhadap noise


**Kelemahan:**
- Waktu training lebih lama dibanding model linear
- Kurang interpretatif dibanding Logistic Regression


#### 6.2.2 Hyperparameter

**Parameter yang digunakan:**
```
n_estimators: 150
max_depth: None
random_state: 42
```

**Hyperparameter Tuning (jika dilakukan):**
- Metode: [Grid Search / Random Search / Bayesian Optimization]
- Best parameters: [...]

#### 6.2.3 Implementasi (Ringkas)
```python
from sklearn.ensemble import RandomForestClassifier


model_rf = RandomForestClassifier(
    n_estimators=150,
    random_state=42
)


model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)


print("Accuracy Random Forest:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))


joblib.dump(model_rf, 'models/random_forest_wine.pkl')
loaded_rf = joblib.load('models/random_forest_wine.pkl')

```

#### 6.2.4 Hasil Model

**[Tuliskan hasil evaluasi, akan dijelaskan detail di Section 7]**

---

### 6.3 Model 3 — Deep Learning Model (WAJIB)

#### 6.3.1 Deskripsi Model

**Nama Model:** [Nama arsitektur, misal: CNN / LSTM / MLP]

** (Centang) Jenis Deep Learning: **
- [✔] Multilayer Perceptron (MLP) - untuk tabular
- [ ] Convolutional Neural Network (CNN) - untuk image
- [ ] Recurrent Neural Network (LSTM/GRU) - untuk sequential/text
- [ ] Transfer Learning - untuk image
- [ ] Transformer-based - untuk NLP
- [ ] Autoencoder - untuk unsupervised
- [ ] Neural Collaborative Filtering - untuk recommender

**Alasan Pemilihan:**  
- Cocok untuk data tabular numerik
- Mampu mempelajari representasi fitur kompleks
- Memenuhi requirement deep learning UAS


#### 6.3.2 Arsitektur Model

**Deskripsi Layer:**

```

| Layer | Deskripsi |
| :--- | :--- |
| **Input** | 13 fitur |
| **Dense 1** | 128 neuron, **ReLU**. |
| **Dropout** | 0.3 |
| **Dense 2** | 64 neuron, **ReLU**. |
| **Dropout** | 0.3 |
| **Output** | 3 neuron, **Softmax** |


Total parameters: [jumlah]
Trainable parameters: [jumlah]
```

#### 6.3.3 Input & Preprocessing Khusus

**Input shape:** 
- Input shape: (13,)
- Target: One-hot encoding
- Fitur telah melalui StandardScaler


#### 6.3.4 Hyperparameter

**Training Configuration:**
```
Optimizer: Adam
Learning rate: default
Loss function: categorical_crossentropy
Metrics: accuracy
Batch size: 16
Epochs: 50 
Validation split: 20%
```

#### 6.3.5 Implementasi (Ringkas)

**Framework:** TensorFlow/Keras / PyTorch
```python
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping


# (Opsional) Matikan warning TensorFlow
tf.get_logger().setLevel('ERROR')


# One-hot encoding
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)


# Model
model_dl = Sequential([
    Input(shape=(X_train.shape[1],)),   # <-- FIX: gunakan Input layer
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(3, activation='softmax')
])


# Compile
model_dl.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# Early stopping
early_stop = EarlyStopping(
    patience=10,
    restore_best_weights=True
)


# Training
history = model_dl.fit(
    X_train, y_train_cat,
    validation_split=0.2,
    epochs=50,
    batch_size=16,
    callbacks=[early_stop],
    verbose=1
)


# Save model
model_dl.save('models/wine_mlp_model.keras')


# Load model
model_dl = load_model('models/wine_mlp_model.keras')

```

#### 6.3.6 Training Process

**Training Time:**  
10.4 detik

**Computational Resource:**  
CPU / GPU, Google Colab

**Training History Visualization:**

[Insert plot loss dan accuracy/metric per epoch]

**Contoh visualisasi yang WAJIB:**
1. **Training & Validation Loss** per epoch
2. **Training & Validation Accuracy/Metric** per epoch

**Analisis Training:**
- Apakah model mengalami overfitting? **Tidak**
  *  Berdasarkan grafik training loss dan validation loss, keduanya menunjukkan pola penurunan yang stabil dan tidak terjadi divergensi yang tajam antara nilai training dan validation. Selain itu, penggunaan teknik Dropout dan Early Stopping membantu mencegah model mempelajari noise secara berlebihan. Perbedaan nilai akurasi antara data training dan validation relatif kecil, sehingga dapat disimpulkan bahwa model tidak mengalami overfitting yang signifikan.
- Apakah model sudah converge? **Ya**
  *  Model menunjukkan kondisi convergence karena nilai loss pada data training dan validation telah mencapai titik stabil dan tidak mengalami penurunan yang signifikan pada epoch-epoch terakhir. Selain itu, mekanisme Early Stopping menghentikan proses training secara otomatis ketika performa validasi tidak lagi meningkat, yang menandakan bahwa model telah mencapai performa optimal.
- Apakah perlu lebih banyak epoch? **Tidak**
  * Penambahan jumlah epoch tidak diperlukan karena model telah mencapai kondisi konvergen sebelum mencapai batas maksimum epoch yang ditentukan. Menambah epoch justru berpotensi meningkatkan risiko overfitting tanpa memberikan peningkatan performa yang berarti pada data uji.

#### 6.3.7 Model Summary
```
[Paste model.summary() output atau rangkuman arsitektur]
```

---

## 7. EVALUATION

### 7.1 Metrik Evaluasi

Metrik yang digunakan untuk klasifikasi multikelas dengan distribusi kelas yang seimbang:
* **Accuracy**
* **Precision**
* **Recall**
* **F1-Score**
* **Confusion Matrix**

### 7.2 Hasil Evaluasi Model

#### 7.2.1 Model 1 (Baseline)

**Metrik:**
```
Metrik:
Accuracy : 0.97
Precision: 0.97
Recall   : 0.96
F1-Score : 0.97

```

**Confusion Matrix / Visualization:**  
[Insert gambar jika ada]

#### 7.2.2 Model 2 (Advanced/ML)

**Metrik:**
```
- Accuracy: 0.85
- Precision: 0.84
- Recall: 0.86
- F1-Score: 0.85
```

**Confusion Matrix / Visualization:**  
[Insert gambar jika ada]

**Feature Importance (jika applicable):**  
[Insert plot feature importance untuk tree-based models]

#### 7.2.3 Model 3 (Deep Learning)

**Metrik:**
```
- Accuracy: 0.89
- Precision: 0.88
- Recall: 0.90
- F1-Score: 0.89
```

**Confusion Matrix / Visualization:**  
[Insert gambar jika ada]

**Training History:**  
[Sudah diinsert di Section 6.3.6]

**Test Set Predictions:**  
[Opsional: tampilkan beberapa contoh prediksi]

### 7.3 Perbandingan Ketiga Model

**Tabel Perbandingan:**

| Model | Accuracy | Precision | Recall | F1-Score | Training Time | Inference Time |
|-------|----------|-----------|--------|----------|---------------|----------------|
| Baseline (Model 1) | 0.75 | 0.73 | 0.76 | 0.74 | 2s | 0.01s |
| Advanced (Model 2) | 0.85 | 0.84 | 0.86 | 0.85 | 30s | 0.05s |
| Deep Learning (Model 3) | 0.89 | 0.88 | 0.90 | 0.89 | 15min | 0.1s |

**Visualisasi Perbandingan:**  
[Insert bar chart atau plot perbandingan metrik]

### 7.4 Analisis Hasil

**Interpretasi:**

1. **Model Terbaik:**  
   [Sebutkan model mana yang terbaik dan mengapa]

2. **Perbandingan dengan Baseline:**  
   [Jelaskan peningkatan performa dari baseline ke model lainnya]

3. **Trade-off:**  
   [Jelaskan trade-off antara performa vs kompleksitas vs waktu training]

4. **Error Analysis:**  
   [Jelaskan jenis kesalahan yang sering terjadi, kasus yang sulit diprediksi]

5. **Overfitting/Underfitting:**  
   [Analisis apakah model mengalami overfitting atau underfitting]

---

## 8. CONCLUSION

### 8.1 Kesimpulan Utama

**Model Terbaik:**  
[Sebutkan model terbaik berdasarkan evaluasi]

**Alasan:**  
[Jelaskan mengapa model tersebut lebih unggul]

**Pencapaian Goals:**  
[Apakah goals di Section 3.2 tercapai? Jelaskan]

### 8.2 Key Insights

**Insight dari Data:**
- [Insight 1]
- [Insight 2]
- [Insight 3]

**Insight dari Modeling:**
- [Insight 1]
- [Insight 2]

### 8.3 Kontribusi Proyek

**Manfaat praktis:**  
[Jelaskan bagaimana proyek ini dapat digunakan di dunia nyata]

**Pembelajaran yang didapat:**  
[Jelaskan apa yang Anda pelajari dari proyek ini]

---

## 9. FUTURE WORK (Opsional)

Saran pengembangan untuk proyek selanjutnya:
** Centang Sesuai dengan saran anda **

**Data:**
- [ ] Mengumpulkan lebih banyak data
- [ ] Menambah variasi data
- [ ] Feature engineering lebih lanjut

**Model:**
- [ ] Mencoba arsitektur DL yang lebih kompleks
- [ ] Hyperparameter tuning lebih ekstensif
- [ ] Ensemble methods (combining models)
- [ ] Transfer learning dengan model yang lebih besar

**Deployment:**
- [ ] Membuat API (Flask/FastAPI)
- [ ] Membuat web application (Streamlit/Gradio)
- [ ] Containerization dengan Docker
- [ ] Deploy ke cloud (Heroku, GCP, AWS)

**Optimization:**
- [ ] Model compression (pruning, quantization)
- [ ] Improving inference speed
- [ ] Reducing model size

---

## 10. REPRODUCIBILITY (WAJIB)

### 10.1 GitHub Repository

**Link Repository:** [URL GitHub Anda]

**Repository harus berisi:**
- ✅ Notebook Jupyter/Colab dengan hasil running
- ✅ Script Python (jika ada)
- ✅ requirements.txt atau environment.yml
- ✅ README.md yang informatif
- ✅ Folder structure yang terorganisir
- ✅ .gitignore (jangan upload dataset besar)

### 10.2 Environment & Dependencies

**Python Version:** [3.8 / 3.9 / 3.10 / 3.11]

**Main Libraries & Versions:**
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
