# SMS Spam Classification using NLP & Linear SVC

Proyek ini adalah implementasi *Machine Learning* untuk mendeteksi pesan SMS Spam dan Ham (pesan normal). Dibangun sebagai tugas mata kuliah *Natural Language Processing* (NLP), proyek ini menggunakan pendekatan ekstraksi fitur TF-IDF dan algoritma klasifikasi *Linear Support Vector Classifier* (Linear SVC) yang dioptimasi menggunakan *Grid Search*.

## 👤 Identitas Pembuat
* **Nama:** Afyu
* **NIM:** 220401191
* **File:** `Tugas_NLP_Afyu(220401191).ipynb`

## 🚀 Fitur & Alur Kerja
Proyek ini mencakup *end-to-end machine learning pipeline*:
1.  **Hardware Check:** Pengecekan ketersediaan GPU (CUDA) untuk akselerasi komputasi (Diuji pada NVIDIA GeForce RTX 3050 Laptop GPU).
2.  **Data Preprocessing:**
    * Loading dataset `spam.csv`.
    * Renaming kolom menjadi `label` dan `text`.
    * Encoding label (Ham=0, Spam=1).
    * Pembersihan *missing values* (NaN).
3.  **Data Splitting:** Pembagian data latih (80%) dan uji (20%) secara *Stratified* untuk menjaga keseimbangan kelas.
4.  **Modeling Pipeline:**
    * **Feature Extraction:** `TfidfVectorizer` (dengan *stopword removal* bahasa Inggris).
    * **Classifier:** `LinearSVC`.
5.  **Hyperparameter Tuning:** Menggunakan `GridSearchCV` untuk mencari kombinasi terbaik:
    * *N-gram range:* Unigram vs Bigram.
    * *Regularization (C):* 0.5, 1, 2.
6.  **Evaluasi:** Mengukur performa menggunakan Akurasi, Precision, Recall, F1-Score, dan Confusion Matrix.

## 🛠️ Tech Stack & Library
Proyek ini dibuat menggunakan Python dengan *library* berikut:
* **Pandas & NumPy:** Manipulasi dan analisis data tabel/array.
* **Scikit-Learn:** Pembuatan model, pipeline, dan evaluasi metrik.
* **Matplotlib & Seaborn:** Visualisasi data (Confusion Matrix Heatmap).
* **PyTorch:** (Opsional) Digunakan dalam notebook untuk pengecekan kompatibilitas CUDA/GPU.

## 📊 Hasil Evaluasi
Berdasarkan proses *Hyperparameter Tuning*, model terbaik didapatkan dengan konfigurasi:
* **Parameter Terbaik:** `{'clf__C': 2, 'tfidf__ngram_range': (1, 2)}` (Bigram).
* **Akurasi Data Uji:** **98.65%**

### Detail Classification Report:
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **Ham** | 0.99 | 1.00 | 0.99 | 966 |
| **Spam**| 0.99 | 0.91 | 0.95 | 149 |

*Model memiliki presisi yang sangat tinggi (0.99) untuk Spam, yang berarti sangat sedikit pesan normal yang salah dikategorikan sebagai spam.*

## 💻 Cara Menjalankan
1.  Pastikan Python dan Jupyter Notebook sudah terinstal.
2.  Install library yang dibutuhkan:
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn torch
    ```
3.  Letakkan file dataset `spam (1).csv` dalam satu folder dengan notebook.
4.  Jalankan `Tugas_NLP_Afyu(220401191).ipynb`.

## 📝 Contoh Prediksi
Model telah diuji secara manual dengan input teks baru:
* *"Congratulations! You have won a free ticket. Call now!"* -> **[SPAM]**
* *"Bro nanti malam jadi ngumpul di kampus?"* -> **[HAM]**
* *"URGENT! You have won 1000$ cash..."* -> **[SPAM]**
