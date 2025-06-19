import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. Set path ke dataset
dataset_path = r"D:\SMESTER 4\KECERDASAN BUATAN\UAS\flowers-recognition"
image_size = (100, 100)  # biar ringan & cepat proses

# 2. Load gambar dan label
X = []
y = []

for label in os.listdir(dataset_path):
    folder = os.path.join(dataset_path, label)
    if os.path.isdir(folder):
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            img = cv2.imread(file_path)
            if img is not None:
                img = cv2.resize(img, image_size)
                img_flat = img.flatten()
                X.append(img_flat)
                y.append(label)

# 3. Ubah jadi array NumPy
X = np.array(X)
y = np.array(y)

# 4. Split data jadi train & test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 5. Latih model KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# 6. Evaluasi akurasi
y_pred = knn.predict(X_test)
print("Akurasi model:", accuracy_score(y_test, y_pred))

# 7. Pilih gambar acak untuk visualisasi prediksi
all_images = [
    (label, os.path.join(dataset_path, label, img))
    for label in os.listdir(dataset_path)
    for img in os.listdir(os.path.join(dataset_path, label))
]

true_label, img_path = random.choice(all_images)
img = cv2.imread(img_path)
img_resized = cv2.resize(img, image_size)
img_flatten = img_resized.flatten().reshape(1, -1)

predicted_label = knn.predict(img_flatten)[0]

# 8. Tampilkan gambar & prediksi
plt.imshow(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
plt.title(f"Asli: {true_label} | Prediksi: {predicted_label}")
plt.axis("off")
plt.show()
