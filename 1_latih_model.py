import joblib
from sklearn.tree import DecisionTreeClassifier
import numpy as np

print("Sedang melatih model AI...")

# Data latih: [IPK, Total SKS]
X_train = np.array([
    [3.5, 144],  # Lulus
    [2.0, 100],  # Tidak Lulus
    [3.2, 140],  # Lulus
    [2.5, 110],  # Tidak Lulus
    [3.8, 150],  # Lulus
    [1.8, 90],   # Tidak Lulus
    [3.0, 130],  # Lulus
    [2.2, 105],  # Tidak Lulus
    [3.6, 145],  # Lulus
    [2.8, 120],  # Lulus
])

# Label: 1 = Lulus, 0 = Tidak Lulus
y_train = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 1])

# Cek jumlah data sama (biar tidak error)
assert len(X_train) == len(y_train), "Jumlah data dan label harus sama!"
print(f"Jumlah data latih: {len(X_train)}")

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Simpan model (Deployment)
joblib.dump(model, 'model_kelulusan.pkl')
print("Sukses! Model disimpan sebagai 'model_kelulusan.pkl'")