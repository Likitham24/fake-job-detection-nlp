# 🕵️ Fake Job Detection using NLP & Machine Learning

## 📌 Project Overview
This project detects fraudulent job postings using Natural Language Processing (NLP) and Machine Learning.

The dataset contains 17,880 job postings with 18 features.  
The target variable is `fraudulent` (0 = Real, 1 = Fake).

---

## 🧠 Problem Statement
Online job platforms contain fraudulent job postings.  
The goal of this project is to build a classification model that detects fake job listings.

---

## 🛠 Technologies Used
- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorization
- Logistic Regression

---

## 🔄 Project Pipeline

1. Data Loading
2. Text Preprocessing
   - Lowercase conversion
   - Remove special characters
3. Text Combination
4. TF-IDF Vectorization
5. Train-Test Split (80-20)
6. Logistic Regression Model
7. Class Imbalance Handling using `class_weight="balanced"`
8. Model Evaluation (Precision, Recall, F1-score)

---

## 📊 Model Performance

### Before Handling Imbalance
- Accuracy: 97%
- Recall (Fake Jobs): 45%

### After Using Balanced Model
- Accuracy: 96%
- Recall (Fake Jobs): 88%

The model significantly improved fraud detection performance by addressing class imbalance.

---

## 📈 Key Learning
- Handling imbalanced datasets is critical
- Accuracy is not enough for fraud detection
- Recall is important for minority class problems

---

## 🚀 Future Improvements
- Try Random Forest / XGBoost
- Use SMOTE for advanced imbalance handling
- Deploy as a web application

---

## 👩‍💻 Author
Likitha M