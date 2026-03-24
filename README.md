# Loan Approval Prediction System

> An ML-powered web application for automated loan approval predictions using XGBoost classification

## 🎯 Business Context

### Content
A loan application captures essential borrower financial information that lenders use to make approval decisions. Automating this analysis improves bank efficiency and decision consistency.

### Problem Statement
Loan officers at SZE Bank spend significant time manually reviewing and filtering applications. This process is:
- **Time-consuming**: Manual review of hundreds of applications
- **Inconsistent**: Subjective criteria across different reviewers
- **Error-prone**: Opportunity for bias in decision-making

### Solution Objective
Build a machine learning model and web application that:
- Automates loan eligibility prediction
- Provides consistent approval decisions
- Reduces processing time significantly
- Maintains transparency through feature importance analysis

---

## 📊 Dataset Overview

**Training Data**: 491 loan applications with 13 financial features
**Test Data**: 123 holdout samples for model validation
**Target Variable**: Loan approval status (Binary classification)
  - Not Approved (Default): 30.1% (148 cases)
  - Approved (Non-Default): 69.9% (343 cases)

### Features
| Feature | Type | Description |
|---------|------|-------------|
| Credit_History | Categorical | Applicant's credit history status (0=poor, 1=good) |
| Loan_Amount_Term | Numeric | Loan tenure in months (default: 360) |
| Loan_Amount | Numeric | Requested loan amount (₹ thousands) |
| ApplicantIncome | Numeric | Primary applicant income (₹ annual) |
| CoapplicantIncome | Numeric | Co-applicant income (₹ annual) |
| Dependents | Categorical | Number of dependents (0, 1, 2, 3+) |
| Education | Categorical | Educational level (Graduate, Undergraduate) |
| Gender | Categorical | Applicant gender |
| Married | Categorical | Marital status |
| Property_Area | Categorical | Property location (Urban, Semi-Urban, Rural) |
| Employment_Type | Categorical | Self-employed or not |
| Loan_ID | Identifier | Unique application ID (removed from model) |

---

## 🤖 Model Architecture

### Algorithm: XGBoost (Extreme Gradient Boosting)
- **Type**: Gradient-boosted decision tree ensemble
- **Framework**: scikit-learn compatible implementation
- **Optimization Method**: GridSearchCV with 5-fold cross-validation

### Hyperparameters
```
eta (learning rate):        0.1
max_depth:                  4
min_child_weight:           1
subsample:                  0.8 (row subsampling)
colsample_bytree:           0.8 (column subsampling)
objective:                  binary:logistic
eval_metric:                error
```

### Data Pipeline
1. **Missing Value Imputation**
   - Categorical features: Mode-based imputation with business logic
   - Numeric features: Median imputation by loan status
   
2. **Feature Engineering**
   - One-hot encoding of 7 categorical variables
   - Standardization not applied (XGBoost is tree-based)
   
3. **Data Splitting**
   - Training set: 70% (343 samples)
   - Validation set: 30% (148 samples) 
   - Final test evaluation: 123 holdout samples

---

## 📈 Model Performance

### Validation Metrics
| Metric | Value | Interpretation |
|--------|-------|-----------------|
| **Accuracy** | 70.27% | Correctly classified 7 out of 10 applications |
| **Precision** | 76.85% | Of approved predictions, 76.85% are correct |
| **Recall** | 81.37% | Catches 81.37% of truly approvable applicants |
| **F1-Score** | 0.7905 | Balanced performance measure |

### Confusion Matrix (Validation Set)
```
                Predicted Approved    Predicted Rejected
Actual Approved:    83 (TP)               19 (FN)
Actual Rejected:    25 (FP)               21 (TN)
```

### Classification Details
- **True Positive Rate (Sensitivity)**: 81.37% - Model identifies actual approvals
- **True Negative Rate (Specificity)**: 45.65% - Model identifies actual rejections
- **False Positive Rate**: 54.35% - Conservative approach favors approvals

---

## 🔍 Feature Importance Analysis

### Top 10 Most Influential Features (by information gain)
1. **Credit_History** (32.48%) - Single strongest predictor
2. **Loan_Amount_Term** (15.39%)
3. **Loan_Amount** (12.24%)
4. **ApplicantIncome** (10.89%)
5. **CoapplicantIncome** (8.15%)
6. **Property_Area_Semi-Urban** (6.28%)
7. **Education_Graduate** (5.21%)
8. **Dependents** (4.87%)
9. **Gender_Male** (2.14%)
10. **Married_Yes** (1.35%)

### Key Insights
- **Credit History Dominance**: Previous credit behavior is 3x more important than income
- **Loan Characteristics**: Term and amount together represent 27.63% of decisions
- **Geographic Factor**: Property location influences approval decisions
- **Income Impact**: Total household income (applicant + co-applicant) has 19.04% influence
- **Demographics**: Family status and education account for ~10% combined weight

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+ (tested on 3.11.13)
- pip or conda package manager
- 100MB free disk space

### Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/xpushkal/loan-prediction.git
cd loan-prediction
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the web application**
```bash
python app.py
```

The application will be available at `http://localhost:5000`

### Required Packages
- pandas — Data manipulation
- numpy — Numerical computing
- scikit-learn — Machine learning utilities
- xgboost — Gradient boosting model
- flask — Web framework
- joblib — Model serialization
- matplotlib — Visualization (used in notebook)
- seaborn — Statistical visualization

---

## 📱 Web Application Usage

### Application Features
1. **Interactive Form**: Input applicant details through web interface
2. **Real-time Predictions**: Get instant loan eligibility prediction
3. **Visual Feedback**: Clear indication of approval probability
4. **Responsive Design**: Works on desktop and mobile devices

---

## 📓 Jupyter Notebook

The complete data science workflow is documented in: `notebook/Machine Learning Model Dev.ipynb`

### Notebook Sections
1. **Data Loading** - Import training and test datasets
2. **Exploratory Data Analysis** - 5 visualizations including:
   - Loan approval class distribution
   - Dependent and education impact analysis
   - Income and loan amount distributions
3. **Data Preprocessing** - Missing value handling and feature engineering
4. **Model Training** - XGBoost with hyperparameter tuning
5. **Model Evaluation** - Performance metrics and confusion matrix
6. **Model Visualization** - 6-panel comprehensive dashboard

---

## 📁 Project Structure

```
loan-approval-prediction-main/
├── app.py                          # Flask web application
├── requirements.txt                # Python dependencies
├── runtime.txt                     # Runtime configuration (Heroku)
├── Procfile                        # Deployment configuration
├── README.md                       # This file
├── .gitignore                      # Git ignore rules
├── PROJECT_DASHBOARD.html          # Interactive project dashboard
│
├── data/
│   ├── loan_train.csv             # Training dataset (491 samples)
│   ├── loan_test.csv              # Test dataset (123 samples)
│   └── columns_set.json           # Feature metadata
│
├── notebook/
│   └── Machine Learning Model Dev.ipynb  # Complete ML pipeline
│
├── bin/                           # Model artifacts (generated)
│   └── model.pkl                  # Trained XGBoost model (if saved)
│
├── static/
│   ├── css/
│   │   └── style.css              # Application styling
│   ├── fonts/                     # Web fonts
│   └── images/                    # Static images
│
└── template/
    ├── index.html                 # Home/input form page
    ├── prediction.html            # Prediction result page
    └── error.html                 # Error page
```

---

## 🔧 Advanced Configuration

### Model Retraining
To retrain the model with latest data:
1. Update `data/loan_train.csv` with new samples
2. Run the Jupyter notebook cells sequentially
3. Export the trained model: `joblib.dump(model, 'bin/model.pkl')`
4. Restart the Flask application

### Deployment to Heroku
```bash
heroku create your-app-name
git push heroku main
heroku open
```

---

## 📊 EDA Key Findings

### Class Imbalance
- 69.9% of applications are approved
- 30.1% are rejected

### Missing Value Patterns
- Credit History: 8.76% missing values
- Employment Type: 8.53% missing
- Other features: <2% missing

---

## 📝 Model Card

### Intended Use
- **Primary Use**: Automated loan eligibility screening for decision support
- **Users**: Loan officers and credit decision systems
- **Decision Context**: Should NOT be the sole factor in approval; use alongside human review

### Ethical Considerations
- Model decisions should be transparent and explainable to applicants
- Regular fairness audits recommended for demographic parity
- Human review process essential for edge cases

---

## 📚 References & Resources

### Libraries Used
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Scikit-learn Guide](https://scikit-learn.org/)
- [Flask Framework](https://flask.palletsprojects.com/)
- [Pandas Documentation](https://pandas.pydata.org/)

---

## 📄 License

This project is provided as-is for educational purposes.

---

### 🎨 Web Template Attribution
This application uses HTML and CSS templates from [Colorlib](https://colorlib.com/wp/template/colorlib-regform-7/)

### Last Updated
March 2024 - Comprehensive documentation and interactive dashboard
# loan-prediction
