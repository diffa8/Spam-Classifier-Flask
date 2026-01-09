import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

# --- INI SESUAI HALAMAN 2 PDF ---

# 1. Load Data
data = pd.read_csv('spam.csv')

# 2. Tentukan X dan y
X = data['Message']
y = data['Category']

# 3. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Buat Model (Pipeline Vectorizer + SVM)
# Di PDF vectorizer dan model dipisah, disini kita gabung biar praktis saat deploy
model_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('svm', SVC(kernel='linear', probability=True))
])

# 5. Latih Model
print("Sedang melatih data...")
model_pipeline.fit(X_train, y_train)

# 6. Simpan Model (Sesuai halaman 3 PDF bagian pickle)
with open('model_svm.pkl', 'wb') as f:
    pickle.dump(model_pipeline, f)

print("BERHASIL! File 'model_svm.pkl' sudah dibuat.")