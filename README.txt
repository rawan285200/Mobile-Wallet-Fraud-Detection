
%%writefile README.md

# 💳 Mobile Wallet Fraud Detection

**Machine Learning | FinTech | Fraud Detection | Classification**

An end-to-end Machine Learning project designed to detect fraudulent mobile money transactions using the PaySim dataset.

The project focuses on handling highly imbalanced data, comparing multiple classification algorithms, optimizing decision thresholds based on financial costs, and deploying an interactive Streamlit application.

---

## 📌 Project Overview

Mobile money services such as Vodafone Cash and InstaPay are growing rapidly. However, fraudulent transactions are also increasing.

The goal of this project is to identify fraudulent transactions while minimizing false alarms that may block legitimate customers.

**Target Variable: `isFraud`**

- `0` → Legitimate Transaction
- `1` → Fraudulent Transaction

---

## 📂 Dataset

**PaySim Mobile Money Dataset**

- Source: [Kaggle - PaySim Dataset](https://www.kaggle.com/datasets/ealaxi/paysim1)
- Original Size: Approximately 6.3 million transactions.
- Sample Size: 500,000 transactions.
- Target Variable: `isFraud`

A stratified sample is used to preserve the original fraud distribution while reducing training time.

---

## 🛠️ Technologies & Libraries

- Python
- Pandas & NumPy
- Matplotlib & Seaborn
- Scikit-learn
- XGBoost
- Joblib
- Streamlit

---

## ⚙️ Project Workflow

### 1. Data Loading & Sampling
- Load the PaySim dataset.
- Apply stratified sampling.
- Preserve the original class distribution.

### 2. Exploratory Data Analysis (EDA)
- Analyze transaction distributions.
- Investigate missing values.
- Explore fraud patterns.
- Visualize class imbalance.

### 3. Feature Engineering

Create additional features to capture unusual balance changes:


- `origBalanceChange`
- `destBalanceChange`
- `origBalanceRatio`
- `destBalanceRatio`
- `amount_log`

### 4. Data Splitting

The dataset is divided into three subsets:

| Dataset | Percentage | Purpose |
|---|---:|---|
| Training | 70% | Model training and Grid Search |
| Validation | 15% | Model comparison and threshold optimization |
| Testing | 15% | Final model evaluation |

Stratified splitting is used to maintain class distribution.

### 5. Model Training & Comparison

Four classification algorithms are compared:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. XGBoost

**Hyperparameter Optimization:**
- GridSearchCV
- Stratified K-Fold Cross-Validation
- Average Precision (PR-AUC) scoring

Class imbalance is addressed using class weights and XGBoost's `scale_pos_weight`.

### 6. Model Evaluation

Models are evaluated using:

- PR-AUC (Average Precision)
- ROC-AUC
- Precision
- Recall
- F1-Score
- Confusion Matrix

PR-AUC is emphasized because fraud transactions represent a very small percentage of the dataset.

### 7. Threshold Optimization

Instead of relying on the default threshold of 0.5, the project searches for a threshold that minimizes total financial loss.

**Assumed Cost Matrix:**

| Error Type | Description | Cost |
|---|---|---:|
| False Negative (FN) | Fraud classified as legitimate | 1000 |
| False Positive (FP) | Legitimate transaction classified as fraud | 50 |

**Total Cost Formula:**

`Total Cost = (FN × Cost_FN) + (FP × Cost_FP)`

The optimal threshold is selected using the validation dataset.

*Note: Cost values are illustrative assumptions and should be adjusted according to real business requirements.*

### 8. Model Saving

The best-performing model is saved using Joblib, together with:

- Model name
- Optimized threshold
- Cost parameters
- Feature names

### 9. Streamlit Deployment

An interactive application provides:

- Single transaction prediction.
- CSV file upload.
- Fraud probability scoring.
- Adjustable classification threshold.
- Fraud and legitimate transaction predictions.
- False Positive and False Negative analysis when actual labels are available.
- Total estimated financial cost.
- Downloadable prediction results.

---

## 📊 Project Outputs

- Model comparison results.
- Best hyperparameters for each algorithm.
- Final PR-AUC and ROC-AUC scores.
- Confusion matrix.
- Optimal classification threshold.
- Cost analysis.
- Saved best model.
- Interactive Streamlit application.

*Actual performance metrics will be added after running the experiments.*

---

## 📁 Project Structure

```text
Mobile-Wallet-Fraud-Detection/
│
├── fraud_detection.ipynb
│
├── app.py
│
├── data/
│   └── paysim.csv
│
├── models/
│   ├── best_fraud_model.pkl
│   ├── model_comparison.csv
│   └── threshold_results.csv
│
├── requirements.txt
│
└── README.md
```

---

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone <your-repository-link>
cd Mobile-Wallet-Fraud-Detection
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the Dataset

Download the PaySim dataset from Kaggle and place the CSV file in the `data/` directory.

### 4. Run the Notebook

Open `fraud_detection.ipynb` and execute the cells sequentially to:

- Preprocess the data.
- Train and compare models.
- Optimize the threshold.
- Save the best model.

### 5. Run the Streamlit Application

```bash
 Python -m streamlit run app.py
```

Open the local URL provided by Streamlit to interact with the application.

---

## 🎯 Key Learning Outcomes

- Working with large-scale transaction datasets.
- Handling extreme class imbalance.
- Feature engineering for financial transactions.
- Hyperparameter tuning using GridSearchCV.
- Comparing machine learning classification algorithms.
- Evaluating models using Precision-Recall metrics.
- Cost-sensitive threshold optimization.
- Saving and deploying machine learning models.
- Building interactive ML applications using Streamlit.

---

## ⚠️ Important Notes

- The dataset is simulated and does not contain actual customer transactions.
- A stratified sample of 1000,000 transactions is used instead of the full dataset.
- The selected threshold is based on assumed financial costs.
- This project is an educational prototype and is not intended for production use.

---

## 👩‍💻 Author

**ENG_RAWAN_MOSTAFA_ABD_EL-RAZIK**

Machine Learning | Data Science | FinTech
