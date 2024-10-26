## Definisi Computer Vision

Computer vision itu kayak bikin komputer bisa "lihat" dan paham gambar atau video, gitu. Jadi, dia bisa ngedeteksi objek, ngenalin wajah, atau bahkan memahami scene. Misalnya, pas kamu pakai filter di Instagram, itu hasil dari computer vision yang ngedit gambar secara otomatis. Intinya, bikin teknologi bisa ngeliat dan ngerti dunia kayak kita!

## Explore bagaimana cara pake data

Untuk eksplorasi data di computer vision, kamu bisa ikutin langkah-langkah ini:

1. **Kumpulin Data**: Cari dataset gambar yang relevan. Banyak yang bisa diunduh gratis, misalnya dari Kaggle atau Google Dataset Search.

2. **Preprocessing**: Bersihin data dengan cara mengubah ukuran, normalisasi, atau augmentasi (misalnya, rotasi atau flipping) biar lebih beragam.

3. **Modeling**: Pilih model yang mau dipakai, kayak CNN (Convolutional Neural Network) yang umum buat tugas-tugas computer vision. Bisa pakai framework seperti TensorFlow atau PyTorch.

4. **Pelatihan**: Latih model dengan data yang udah diproses. Pastikan kamu bagi data jadi data latih dan data uji.

5. **Evaluasi**: Uji model dengan data uji untuk liat seberapa baik performanya. Gunakan metrik seperti akurasi atau F1 score.

6. **Eksperimen**: Coba variasi model, arsitektur, atau hyperparameter untuk ningkatin performa.

7. **Aplikasi**: Setelah model siap, coba implementasi ke aplikasi nyata, kayak deteksi objek, klasifikasi gambar, atau bahkan video analisis.

8. **Visualisasi**: Pakai tools buat visualisasi hasil, biar lebih mudah paham outputnya.

# Tulis code buat latihan 

Berikut adalah contoh dasar untuk menulis kode latihan menggunakan Python dan TensorFlow/Keras di computer vision. Kita akan menggunakan dataset MNIST (angka tulisan tangan) sebagai contoh.

### 1. Instalasi Library

Pastikan kamu sudah menginstal TensorFlow. Jika belum, kamu bisa install dengan:

```bash
pip install tensorflow
```

### 2. Import Library

```python
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
```

### 3. Load Data

```python
# Load dataset MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalisasi data
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Ubah bentuk data menjadi (jumlah_sample, tinggi, lebar, channel)
x_train = x_train.reshape((60000, 28, 28, 1))
x_test = x_test.reshape((10000, 28, 28, 1))
```

### 4. Membangun Model

```python
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
```

### 5. Kompilasi Model

```python
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

### 6. Latih Model

```python
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
```

### 7. Evaluasi Model

```python
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc}')
```

### 8. Visualisasi Hasil (Opsional)

```python
# Contoh visualisasi beberapa prediksi
predictions = model.predict(x_test)

plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title(f'Prediksi: {predictions[i].argmax()}')
    plt.axis('off')
plt.show()
```

Dengan langkah-langkah di atas, kamu bisa mulai latihan computer vision menggunakan dataset sederhana. Coba modifikasi arsitektur model atau parameter lainnya untuk lihat perubahannya!

