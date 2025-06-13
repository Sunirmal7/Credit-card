# 💳 Credit Card Fraud Detection

This is Task 2 of my Machine Learning internship at **CodSoft**.  
The goal is to build a machine learning model that can accurately detect fraudulent credit card transactions using transaction data.

---

## 📁 Project Structure

Credit_Card_Fraud_Detection/
│
├── fraud_detection.py
├── fraud_scaler.pkl
├── feature_names.json
├── demo_presentation.pptx
├── requirements.txt
├── README.md
└── data/
└── README.md (with Google Drive dataset links)

markdown
Copy
Edit

---

## 📊 Dataset

The dataset includes thousands of real credit card transactions, with a binary column `is_fraud` indicating fraud status.

- **Source:** [Kaggle - Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection)

Due to file size constraints, dataset files are hosted externally:

📥 [Download fraudTrain.csv](https://drive.google.com/file/d/XXXX/view?usp=sharing)  
📥 [Download fraudTest.csv](https://drive.google.com/file/d/YYYY/view?usp=sharing)

Place both files inside the `/data` folder before running the script.

---

## ⚙️ How the Project Works

1. ✅ **Data Loading & Cleaning**
   - Combine train and test datasets
   - Drop unnecessary columns (names, address, etc.)

2. ✅ **Feature Engineering**
   - Select only numeric columns
   - Apply Standard Scaler for normalization

3. ✅ **Modeling**
   - Train `RandomForestClassifier` on the processed dataset
   - Stratified train-test split ensures class balance

4. ✅ **Evaluation**
   - Confusion Matrix & Classification Report
   - Balanced performance on precision, recall, and F1-score

5. ✅ **Output**
   - Saved model (`fraud_model.pkl`)
   - Scaler object (`fraud_scaler.pkl`)
   - Used feature list (`feature_names.json`)

---

## 🧪 Installation & Running

### 📌 Prerequisites

- Python 3.8+
- Install required packages:
  ```bash
  pip install -r requirements.txt
▶️ Run the Script
bash
Copy
Edit
python fraud_detection.py
📊 Sample Output
bash
Copy
Edit
Confusion Matrix:
[[18976    14]
 [   33   264]]

Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     18990
           1       0.95      0.89      0.92       297
📦 Output Files
fraud_model.pkl → Trained model

fraud_scaler.pkl → Scaler for input features

feature_names.json → List of input feature names

