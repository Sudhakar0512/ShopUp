# Loan Default Prediction: Data Science Hands-On Solution

## Overview

This project develops an end-to-end machine learning solution to predict loan defaults using two datasets: `payment_default.csv` (client-level data) and `payment_history.csv` (payment behavior over time). The solution includes data loading, exploratory data analysis (EDA), data cleaning, feature engineering, model development, evaluation, and a scoring function for predicting defaults on new data. Class imbalance is addressed using class weights, and the final model is saved for deployment.

The code is implemented in a Jupyter Notebook using Python, leveraging libraries like `pandas`, `scikit-learn`, `xgboost`, `matplotlib`, and `seaborn`. This document explains the workflow, outputs, and instructions for running and testing the solution.

---

## Problem Statement

The goal is to predict whether a client will default on a loan (`default = 1`) or not (`default = 0`) based on demographic information (e.g., gender, education, marital status), credit amount, and historical payment behavior. The model must handle class imbalance, as defaults are typically less frequent, and provide robust predictions on new data.

Key objectives:
- Perform EDA to understand data distributions, relationships, and potential issues.
- Clean and preprocess the data to ensure high-quality input for modeling.
- Engineer features to capture meaningful patterns in payment behavior.
- Train and evaluate multiple machine learning models, selecting the best performer.
- Develop a scoring function to predict defaults on new data.
- Save the model and related objects for deployment.

---

## Datasets

The solution uses two input datasets:

1. **`payment_default.csv`**:
   - Contains client-level data.
   - Required columns:
     - `client_id`: Unique identifier for each client.
     - `credit_given`: Amount of credit provided.
     - `gender`: Client's gender.
     - `education`: Client's education level.
     - `marital_status`: Client's marital status.
     - `default`: Target variable (1 = default, 0 = no default).

2. **`payment_history.csv`**:
   - Contains historical payment data for each client.
   - Required columns:
     - `client_id`: Unique identifier linking to `payment_default.csv`.
     - `payment_status`: Payment status (-1 = on time, 1–9 = months delayed).
     - `bill_amt`: Bill amount for the month.
     - `paid_amt`: Amount paid for the month.
     - `month`: Month of the payment record.

Test datasets (`TestFiles/test_payment_default.csv` and `TestFiles/test_payment_history.csv`) are used to validate the scoring function and should follow the same structure (with `default` optional for evaluation).

---

## Methodology

The solution follows a structured pipeline with seven key steps:

### Step 1: Import Libraries
- Libraries used:
  - `pandas`, `numpy`: Data manipulation.
  - `matplotlib`, `seaborn`: Data visualization.
  - `scikit-learn`: Preprocessing, model training, and evaluation.
  - `xgboost`: XGBoost classifier for ensemble modeling.
  - `warnings`: Suppress warning messages.
- Random seed set to 42 for reproducibility.

### Step 2: Load and Merge Data
- Load `payment_default.csv` and `payment_history.csv`.
- Validate required columns to ensure data integrity.
- Aggregate payment history by `client_id` to compute summary statistics (e.g., mean, max, min, std) for `payment_status`, `bill_amt`, `paid_amt`, and `month`.
- Merge the aggregated history with client data using a left join on `client_id`.
- Output: A merged DataFrame with client-level features and aggregated payment history.

### Step 3: Exploratory Data Analysis (EDA)
- **Summary Statistics**: Examine distributions of numerical features.
- **Missing Values**: Identify and quantify missing data.
- **Target Distribution**: Visualize the distribution of `default` and compute the default rate.
- **Categorical Analysis**: Plot default rates by `gender`, `education`, and `marital_status`.
- **Payment Status**: Analyze the distribution of `payment_status` and its variation by `month`.
- **Correlations**: Generate a correlation matrix for numerical features.
- **Outliers**: Detect outliers in `credit_given` and cap at the 99th percentile to mitigate their impact.
- Output: Visualizations (e.g., count plots, histograms, heatmaps) and insights into data patterns.

### Step 4: Data Cleaning
- Impute missing values in history-related columns (e.g., `payment_status_mean`, `bill_amt_sum`) using the median.
- Verify that no missing values remain.
- Ensure data consistency (e.g., correct data types, no duplicates).
- Output: A clean DataFrame ready for feature engineering.

### Step 5: Feature Engineering
- Create new features:
  - `repayment_ratio`: Ratio of total paid amount to total bill amount (`paid_amt_sum / (bill_amt_sum + 1e-6)`).
  - `months_since_earliest`: Time span between the latest and earliest payment months.
- Encode categorical variables (`gender`, `education`, `marital_status`) using one-hot encoding with `drop_first=True` to avoid multicollinearity.
- Drop unnecessary columns (`client_id`, `default`) to create the feature matrix.
- Save the list of feature columns (`training_features`) for use in the scoring function.
- Output: Feature matrix (`features_df`) and target vector (`target`).

### Step 6: Model Development
- Split data into training (80%) and testing (20%) sets with stratification to maintain class balance.
- Compute class weights for XGBoost (`scale Facet (`scale_pos_weight = neg/pos`) to handle class imbalance.
- Scale features using `StandardScaler`.
- Train three models:
  - Logistic Regression (with `class_weight='balanced'`).
  - Random Forest (with `class_weight='balanced'`).
  - XGBoost (with `scale_pos_weight`).
- Evaluate models using 5-fold cross-validation (scoring: ROC AUC) and test set metrics (AUC, precision, recall, F1).
- Select the best model based on test AUC.
- Perform hyperparameter tuning for XGBoost (if selected) using `GridSearchCV` over `n_estimators`, `max_depth`, and `learning_rate`.
- Tune the decision threshold to maximize F1 score.
- Visualize feature importance for tree-based models (Random Forest or XGBoost).
- Output: Trained `best_model`, performance metrics, and feature importance plot.

### Step 7: Scoring Function
- Implement `score_new_data` to preprocess new data and generate predictions:
  - Load and validate input files.
  - Aggregate payment history and merge with client data.
  - Clean data (impute missing values using specified strategy: `zero`, `mean`, or `median`).
  - Engineer features (`repayment_ratio`, `months_since_earliest`).
  - Encode categorical variables and align features with `training_features`.
  - Scale features using the saved `scaler`.
  - Predict probabilities and binary outcomes using the saved `best_model` and specified `threshold`.
- Output: A DataFrame with `client_id`, `probability_of_default`, and `default_indicator`.
- Test the function on `TestFiles/test_payment_default.csv` and `TestFiles/test_payment_history.csv`.

---

## Outputs

### Saved Files
The following files are saved using `joblib.dump`:

1. **`best_model.pkl`**:
   - Contains the trained best model (e.g., tuned XGBoost classifier).
   - Used by the scoring function to make predictions.

2. **`scaler.pkl`**:
   - Contains the fitted `StandardScaler` object with mean and standard deviation from training data.
   - Used to scale new data consistently.

3. **`training_features.pkl`**:
   - Contains a list of feature column names used during training.
   - Ensures feature alignment in the scoring function.

### Other Outputs
- **Plots**: Visualizations from EDA (e.g., target distribution, correlation matrix, feature importance) are displayed but not saved unless explicitly modified (e.g., using `plt.savefig`).
- **Predictions**: The `score_new_data` function outputs a DataFrame with predictions. This can be saved to `predictions.csv` by adding `output.to_csv('predictions.csv')`.
- **Metrics**: Cross-validation and test set performance metrics (AUC, precision, recall, F1) are printed for each model.

---

## Installation and Setup

### Prerequisites
- Python 3.8+
- Required libraries:
  ```bash
  pip install pandas numpy matplotlib seaborn scikit-learn xgboost
