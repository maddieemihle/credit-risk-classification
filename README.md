# Credit Risk Analysis Report 

## Introduction 
The purpose of this analysis is to build and evaluate machine learning models to predict loan risk: healthy ("low risk") or "high-risk". The goal is to help make informed loan approval decisions based on historical data.

## Overview of the Analysis

The dataset contains financial information about loans, such as loan amount, interest rate, and borrower details. The target variable, `loan_status`, is binary, with `0` representing healthy loans and `1` representing high-risk loans. The dataset shows a significant class imbalance, with more healthy loans than high-risk loans.

The stages of the machine learning process included:
1. Splitting the data into training and testing sets.
2. Standardizing the features to improve model performance.
3. Training and evaluating two machine learning models:
    * **Logistic Regression:** A linear model for binary classification.
    * **Random Forest:** A non-linear ensemble model for comparison.
4. Evaluating the models using metrics such as accuracy, precision, recall, and F1-score.

By comparing Logistic Regression with Random Forest, we aimed to determine which model performs better for this classification task.

### Methods:  
**1. Data Preprocessing:**
* Split the dataset into features (`X`) and labels (`y`).
* Use `train_test_split` for training and testing sets.
* Standardize features with `StandardScaler`.

**2. Machine Learning Models:**
* Two models were used for comparison: Logistic Regression & Random Forest 

**3. Model Evaluation:**
* `confusion_matrix`: Used to evaluate the number of true positives, true negatives, false positives, and false negatives for both models.
* `classification_report`: Generated to calculate precision, recall, F1-score, and accuracy for both classes (`0` for healthy loans and `1` for high-risk loans).

Jupyter Notebook: [Here](https://github.com/maddieemihle/credit-risk-classification/blob/main/Credit_Risk/credit_risk_classification.ipynb)

## Results
### Machine Learning Model 1: Logistic Regression 
**Confusion Matrix Analysis:**
* Healthy Loans (`0`):
    * True Positives: 18,658
    * False Negatives: 107
    * The model performs exceptionally well in predicting healthy loans, with very few misclassifications.
* High-Risk Loans (`1`):
    * True Positives: 582
    * False Negatives: 37
    * The model is effective at identifying high-risk loans, though it misses a small number of them.

**Classification Report Analysis:**
* Accuracy: 99%
* Precision:
    * `low_risk` (0): 1.00 
    * `high_risk` (1): 0.84 
* Recall:
    * `low_risk` (0): 0.99 
    * `high_risk` (1): 0.94
* F1-Score:
    * `low_risk` (0): 1.00 
    * `high_risk` (1): 0.89 

### Machine Learning Model 2: Random Forest 
**Confusion Matrix Analysis:**
* Healthy Loans (`0`):
    * True Positives: 18,666
    * False Negatives: 99
    * The model performs exceptionally well in predicting healthy loans, with very few misclassifications.
* High-Risk Loans (`1`):
    * True Positives: 553
    * False Negatives: 66
    * The model is effective at identifying high-risk loans but misses more high-risk loans compared to Logistic Regression.

**Classification Report Analysis:**
* Accuracy: 99%
* Precision:
    * `low_risk` (0): 1.00 
    * `high_risk` (1): 0.85 
* Recall:
    * `low_risk` (0): 0.99
    * `high_risk` (1): 0.89 
* F1-Score:
    * `low_risk` (0): 1.00 
    * `high_risk` (1): 0.87 

## Summary
Both models performed exceptionally well, achieving high accuracy and strong metrics for both `0` (healthy loans) and `1` (high-risk loans). However, there are some differences:

* Logistic Regression:
    * Achieved slightly higher recall (94%) for high-risk loans, meaning it identified more high-risk loans correctly.
    * Slightly lower precision (84%) for high-risk loans, meaning it had more false positives compared to Random Forest.
* Random Forest:
    * Achieved slightly higher precision (85%) for high-risk loans, meaning it had fewer false positives.
    * Slightly lower recall (89%) for high-risk loans, meaning it missed more high-risk loans compared to Logistic Regression.

Again, the choice of the model depends on the business objective: 
* If minimizing false negatives (missing high-risk loans) is more critical, Logistic Regression is recommended due to its higher recall for high-risk loans.
* If minimizing false positives (flagging healthy loans as high-risk) is more important, Random Forest is recommended due to its higher precision for high-risk loans.

Overall, the Logistic Regression is slightly better suited in credit risk classification. Its high accuracy and strong recall for high-risk loans make it a reliable tool for identifying risky loans. It should be noted that further improvements could focus on reducing false positives for high-risk loans to enhance precision. This would ensure fewer healthy loans are incorrectly flagged as high-risk, improving the model's overall utility for decision-making. However, the Random Forest model provides a strong alternative with comparable performance. 

## Tools & Technologies Used 
* Python
* Pandas
* Scikit-learn: 
    * `train_test_split`
    * `LogisticRegression`
    * `RandomForestClassifier` 
    * `StandardScaler`
    * `confusion_matrix`
    * `classification_report`
* Jupyter Notebook 
* Visual Studio Code 

## References 
Data for this dataset was generated by _edX Boot Camps LLC_, and is intended for educational purposes only.