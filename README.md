# Demand Forecasting Decision Support System

## Overview
This project implements a data-driven decision support system for demand forecasting in small business environments.  
It combines time series forecasting with a rule-based decision layer to evaluate not only prediction accuracy but also decision quality.

---

## Research Objective
To investigate whether improving predictive accuracy leads to better decision-making in a business context.

---

## Dataset
- Kaggle: Store Sales – Time Series Forecasting
- Filtered to a single store and product family (GROCERY I)

---

## Methodology

### 1. Data Preprocessing
- Date parsing and sorting
- Missing value handling
- Filtering to single time series

### 2. Feature Engineering
- Time-based features: day, month, day of week
- Lag features: lag_7, lag_14, lag_30

### 3. Model
- Random Forest Regressor
- Hyperparameters:
  - n_estimators = 200
  - max_depth = 10

### 4. Decision Support System
A rule-based system converts predictions into business actions:

- Low demand → Reduce stock  
- Medium demand → Maintain stock  
- High demand → Increase stock  

Thresholds are based on data percentiles.

---

## Evaluation Metrics

### Prediction Level
- Mean Absolute Error (MAE)

### Decision Level
- Decision Accuracy (agreement between predicted and true decisions)

---

## Results
- MAE: ~423.48  
- Decision Accuracy: ~0.70  

---

## Key Insight
Improving predictive accuracy does not necessarily improve decision quality.  
This highlights the importance of evaluating machine learning systems at the decision level, not only the prediction level.

---

## Tools
- Python
- Pandas
- Scikit-learn
