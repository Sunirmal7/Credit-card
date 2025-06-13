import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import json

print("Script started...")

train_path = r"C:\Users\KIIT\OneDrive\Desktop\Progs\CODSOFT\Credit Card\fraudTrain.csv"
test_path = r"C:\Users\KIIT\OneDrive\Desktop\Progs\CODSOFT\Credit Card\fraudTest.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

df = pd.concat([train_df, test_df], ignore_index=True)
print("Data loaded and combined.")

df.columns = df.columns.str.strip()

target_col = "is_fraud"
if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' not found in dataset.")
print("Target column verified.")


non_features = ['Unnamed: 0', 'first', 'last', 'street', 'city', 'state', 'zip',
                'trans_date_trans_time', 'dob', 'job', 'merchant', 'category', 
                'trans_num', 'cc_num']
df = df.drop(columns=[col for col in non_features if col in df.columns], errors='ignore')
print("Irrelevant columns dropped.")


df = df.select_dtypes(include=['int64', 'float64'])
print("Numeric columns selected.")


X = df.drop(target_col, axis=1)
y = df[target_col]
feature_names = X.columns.tolist()
print("Features and labels separated.")


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Features scaled.")


X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print("Train-test split done.")


print("⏳ Training model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print(" Model training completed.")


y_pred = model.predict(X_test)
print("\n Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n Classification Report:\n", classification_report(y_test, y_pred))


joblib.dump(model, r"C:\Users\KIIT\OneDrive\Desktop\Progs\CODSOFT\Credit Card\fraud_model.pkl")
joblib.dump(scaler, r"C:\Users\KIIT\OneDrive\Desktop\Progs\CODSOFT\Credit Card\fraud_scaler.pkl")
with open(r"C:\Users\KIIT\OneDrive\Desktop\Progs\CODSOFT\Credit Card\feature_names.json", "w") as f:
    json.dump(feature_names, f)

print("\n Model, scaler, and feature names saved successfully.")
print("🏁 Script completed.")
