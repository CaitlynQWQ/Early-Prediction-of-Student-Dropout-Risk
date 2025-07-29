# 🎓 Early Prediction of Student Dropout Risk Using Machine Learning

## 📘 Project Overview

This project aims to mitigate academic failure in higher education by applying machine learning techniques to **identify students at risk of dropping out early** in their academic journey.

We use a real-world dataset from a higher education institution, containing **enrollment-time information** including academic history, demographics, and socio-economic status. The task is framed as a **binary classification problem** with a significant class imbalance, which is addressed using **synthetic oversampling techniques**.

Key stakeholders include **educators, administrators, and policy-makers**, as early identification enables **targeted interventions** to improve student retention and success rates.

---

## 🧠 Methods & Models

* **Algorithms used**:

  * Decision Tree
  * Random Forest
  * Logistic Regression
  * K-Nearest Neighbors (KNN)
  * AdaBoost
  * Support Vector Machines (SVM)
  * *(XGBoost optional – commented out)*

* **Techniques applied**:

  * Exploratory Data Analysis (EDA)
  * Handling imbalanced data using oversampling (e.g., SMOTE)
  * Feature scaling (`StandardScaler`)
  * Evaluation using:

    * Accuracy
    * Precision
    * Recall
    * F1-Score
    * Classification Report

* **Visualization**: Histograms, correlation heatmaps, confusion matrices

---

## 📊 Sample Code Snippet

```python
df = pd.read_csv('droupout.csv')
plt.hist(df['Admission grade'], bins=30, color='blue', edgecolor='black')
plt.title('Distribution of Admission grade at Enrollment')
plt.xlabel('Admission grade')
plt.ylabel('Frequency')
plt.show()
```

---

## 📂 Files

| File Name             | Description                                          |
| --------------------- | ---------------------------------------------------- |
| `Final Project.ipynb` | Main notebook containing data analysis and ML models |
| `droupout.csv`        | Input dataset (not provided in repo, for privacy)    |

---

## 👩‍💻 Author

*Caitlyn Cai* — Passionate about applying machine learning to education and public interest domains.
