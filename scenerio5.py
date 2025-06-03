import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, top_k_accuracy_score
from sklearn.preprocessing import StandardScaler

BASE_DIR = "D:/Dataset-main/Dataset-main"


def load_data(csv_filename):
    df = pd.read_csv(os.path.join(BASE_DIR, csv_filename))
    X, y = [], []

    for idx, row in df.iterrows():
        try:
            # Görüntü işle
            img_path = os.path.join(BASE_DIR, row['unit1_rgb_1'].replace("./", ""))
            img = Image.open(img_path).resize((64, 64))
            img_array = np.array(img).flatten() / 255.0

            # mmWave power işle
            txt_path = os.path.join(BASE_DIR, row['unit1_pwr_1'].replace("./", ""))
            power_array = np.loadtxt(txt_path)

            # Birleştir
            features = np.concatenate((img_array, power_array))
            X.append(features)
            y.append(row['beam_index_1'])
        except Exception as e:
            print(f"Hata: {e} → Satır atlandı: {idx}")

    return np.array(X), np.array(y)

X_train, y_train = load_data("scenario5_dev_train.csv")
X_val, y_val = load_data("scenario5_dev_val.csv")
X_test, y_test = load_data("scenario5_dev_test.csv")


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_val_proba = model.predict_proba(X_val)
all_labels_val = np.unique(np.concatenate((y_train, y_val)))

print("\n📊 VALIDASYON SONUÇLARI")
print("Top-1 accuracy:", top_k_accuracy_score(y_val, y_val_proba, k=1, labels=all_labels_val))
print("Top-2 accuracy:", top_k_accuracy_score(y_val, y_val_proba, k=2, labels=all_labels_val))
print("Top-3 accuracy:", top_k_accuracy_score(y_val, y_val_proba, k=3, labels=all_labels_val))


y_test_proba = model.predict_proba(X_test)
all_labels_test = np.unique(np.concatenate((y_train, y_test)))

print("\n🧪 TEST SONUÇLARI")
print("Top-1 accuracy:", top_k_accuracy_score(y_test, y_test_proba, k=1, labels=all_labels_test))
print("Top-1 accuracy:", top_k_accuracy_score(y_test, y_test_proba, k=2, labels=all_labels_test))
print("Top-3 accuracy:", top_k_accuracy_score(y_test, y_test_proba, k=3, labels=all_labels_test))


y_test_pred = model.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_test_pred))

y_pred = model.predict(X_test)


comparison_df = pd.DataFrame({
    "Gerçek Label": y_test,
    "Tahmin Label": y_pred
})

print(comparison_df.head(20))



