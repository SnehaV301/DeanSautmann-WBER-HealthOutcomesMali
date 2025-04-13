# 🏥 Healthcare Access Prediction using Machine Learning
This project uses machine learning to predict whether individuals will utilize free healthcare services based on health and household survey data. It aims to support healthcare planning and policy decisions by identifying key factors that influence treatment-seeking behavior.

-------
## Objective
To build and deploy a predictive model that determines the likelihood of individuals utilizing free healthcare if offered, using demographic, health, and socio-economic data.

----
## 📊 Problem Statement
Access to healthcare is a crucial determinant of public health. Many individuals in low-resource settings may not utilize free healthcare services due to various barriers. This project aims to predict whether a patient will **utilize free healthcare** (`free_care_check_`) based on features like treatment history, water source, income proxy (LogAssets), and more.

------
## 🧠 ML Solution
We built a classification model using a structured dataset of patient records. The model was trained using the following features:
### 🔢 Input Features:
- The degree of treatment availed in the past
- Literacy of female adult
- Availability of vaccine card, helath program or health worker's support
- Correctness of ORT recipe or breast feeding duration
### 🎯 Target Variable:
- `free_care_check_` (Binary: 0 = Did not utilize, 1 = Utilized)
---
This project involved many pre-processing steps such as handling missing values and detecting outliers to ensure data quality.
**Feature Scaling** was applied using `StandardScaler` to normalize numerical features and improve model performance.
After preprocessing, feature selection was performed using correlation analysis and many other methods like PCA, MI etc. Model-based importance scores also checked to retain only the most relevant features. 
**Model Selection** involved evaluating several classifiers, including:
  - Logistic Regression
  - Random Forest
  - XGBoost
  - Decision Tree<br>
Among them, the XGBoost delivered the best results, achieving an accuracy of 96.33%.
Model performance was assessed using precision, recall, F1-score, and a confusion matrix.
Finally, hyperparameter tuning was conducted using GridSearchCV.
The best-performing model was chosen based on cross-validation accuracy and F1 score.
---
## 🚀 Deployment

The final solution was deployed using **Flask** with a simple HTML frontend (`index.html`) that:
- Accepts user input for the 10 features
- Sends the data to `app.py`
- Returns the prediction (`Will utilize free healthcare: Yes/No`)
---

