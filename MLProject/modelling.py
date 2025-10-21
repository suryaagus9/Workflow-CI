import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# --- Memuat Data ---
DATA_PATH = 'telco_churn_preprocessing.csv'

print("Memulai proses training model...")
try:
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset berhasil dimuat dari '{DATA_PATH}'.")
except FileNotFoundError:
    print(f"Error: File '{DATA_PATH}' tidak ditemukan. Pastikan path sudah benar.")
    exit()

# --- Persiapan Data ---
# Pisahkan fitur (X) dan target (y)
X = df.drop('Churn_Yes', axis=1)
y = df['Churn_Yes']

# Bagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Data berhasil dibagi menjadi data latih dan uji.")

# --- Autologging MLflow ---
with mlflow.start_run():
    # Mengaktifkan autologging untuk library Scikit-learn di dalam run.
    mlflow.sklearn.autolog()

    # Inisialisasi dan latih model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    print("Model berhasil dilatih.")
    
    mlflow.sklearn.log_model(sk_model=model, artifact_path="model")
    print("Model secara eksplisit dicatat ke path artefak 'model'.")

    print(f"\nTraining selesai. Semua metrik dan artefak telah dicatat secara otomatis oleh autolog.")