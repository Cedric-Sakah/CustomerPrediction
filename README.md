# Telecom Churn Prediction

## Overview
This project aims to predict customer churn in the telecom sector using machine learning models. The dataset contains customer information, including demographics, service usage, and billing details. Various models such as Logistic Regression, Decision Tree, SGDClassifier (for online learning), and an Ensemble Voting Classifier are implemented to analyze and predict customer churn trends.

## Features
- Data preprocessing: Handling missing values, encoding categorical variables, and scaling numeric features.
- Outlier detection and handling using IQR.
- Handling class imbalance with SMOTE.
- Feature correlation analysis using heatmaps.
- Training models with Logistic Regression and Decision Trees.
- Concept drift detection by evaluating model performance over three years.
- Online learning with an SGDClassifier for incremental training.
- Building an Ensemble Voting Classifier to improve prediction accuracy.

## Dataset
The dataset contains customer attributes such as:
- Age
- Income
- Monthly usage (minutes & data)
- Support tickets
- Monthly bill & outstanding balance
- Region (North, South, West)
- Churn status (Target variable)

## Installation
### Prerequisites
- Python 3.8+
- Libraries: Pandas, NumPy, Seaborn, Scikit-learn, Imbalanced-learn, Matplotlib, OpenPyXL

```bash
pip install pandas numpy seaborn scikit-learn imbalanced-learn matplotlib openpyxl
```

## Running the Code
1. Ensure `synthetic_telecom_churn_dataset.xlsx` is available in the working directory.
2. Run the script:
   ```bash
   python churn_prediction.py
   ```
3. The model will preprocess the dataset, train multiple classifiers, detect concept drift, and evaluate performance.

## Results & Evaluation
The script provides:
- Model evaluation metrics: Accuracy, Precision, Recall, F1 Score, and ROC-AUC.
- Comparison of model performance over different years to detect concept drift.
- An ensemble model combining multiple classifiers to improve prediction accuracy.

## Future Improvements
- Implement deep learning models for better churn prediction.
- Use additional features such as customer sentiment analysis.
- Explore hyperparameter tuning for optimal model performance.

