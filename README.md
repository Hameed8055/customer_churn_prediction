# Customer Churn Prediction

## 📌 Project Overview
This project focuses on analyzing customer churn data and building a machine learning model to predict whether a customer is likely to churn. The goal is to derive **business insights** from exploratory data analysis (EDA) and then develop a predictive model that can help businesses **retain customers** more effectively.

---

## 🗂 Project Workflow

### 1. Import Libraries
All required Python libraries (NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, etc.) were imported for data analysis, visualization, and machine learning.

### 2. Exploratory Data Analysis (EDA)
- **Churn Distribution**: Understanding the percentage of customers who churned vs retained.  
- **Numerical Features vs Churn**: Analyzing how numerical variables impact churn.  
- **Categorical Features vs Churn**: Studying churn rates across contract type, internet service, payment method, etc.  
- **Correlation Heatmap**: Identifying relationships between numerical features.  
- **Boxplots & Visual Insights**: To spot patterns and outliers.  

### 3. Key Business Insights
- **Contract Type vs Churn**: Customers with month-to-month contracts are more likely to churn.  
- **Internet Service vs Churn**: Fiber optic customers show higher churn rates.  
- **Payment Method vs Churn**: Electronic check users churn more frequently.  

### 4. Feature Engineering & Preprocessing
- Dropped irrelevant columns (e.g., `customerID`).  
- Encoded categorical variables using OneHotEncoding/LabelEncoding.  
- Defined **features (X)** and **target (y)**.  
- Split dataset into **train** and **test sets**.  
- Scaled numerical features for model efficiency.  

### 5. Model Building & Evaluation
- Implemented multiple ML models (e.g., Logistic Regression, Decision Tree, Random Forest, XGBoost).  
- Compared performance metrics (Accuracy, Precision, Recall, F1-score).  
- Final model saved for deployment (`customer_churn_model.pkl`).  

---

## 🛠 Tech Stack
- **Programming Language**: Python  
- **Libraries**: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost, Joblib  
- **Environment**: Jupyter Notebook  

---

## 📊 Results
- Derived **business insights** to help reduce churn.  
- Built a predictive model with high accuracy and balanced performance across precision/recall.  
- Model and preprocessing pipeline are saved for reuse.  

---

## 🚀 How to Run
1. Clone this repository  
   ```bash
   git clone <your-repo-link>
   cd customer-churn-prediction
   ```
2. Install dependencies  
   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebook  
   ```bash
   jupyter notebook
   ```
4. Explore the analysis and predictions.  

---

## 📌 Future Improvements
- Deploy the model using **Streamlit** or **Flask** for interactive predictions.  
- Experiment with **deep learning models**.  
- Add **SHAP/LIME explainability** for model interpretability.  
