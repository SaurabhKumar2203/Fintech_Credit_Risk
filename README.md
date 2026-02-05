# 🏦 Fintech Credit Risk Scoring Model

![Status](https://img.shields.io/badge/Status-Completed-success)
![Tech](https://img.shields.io/badge/Tech-XGBoost%20%7C%20SMOTE%20%7C%20Python-blue)
![Domain](https://img.shields.io/badge/Domain-BFSI%20%26%20Risk-red)

## 📌 Project Overview
Unsecured lending (BNPL/Personal Loans) in India faces a high risk of Non-Performing Assets (NPAs). This project builds a **Credit Risk Scoring Model** to predict borrower defaults.

I utilized **SMOTE (Synthetic Minority Over-sampling Technique)** to handle the massive class imbalance (only 5% defaults) and trained an **XGBoost Classifier** to identify high-risk applicants.

---

## 📸 Model Performance & Explainability

### 1. Confusion Matrix (The "Recall" Challenge)
Finding defaulters in an imbalanced dataset is difficult.

<img width="752" height="712" alt="Screenshot 2026-02-05 171240" src="https://github.com/user-attachments/assets/758b8664-d2c8-4391-b054-474334c9e574" />
*(Figure 1: The model successfully identified 105 high-risk defaulters (Bottom-Right) who would have otherwise passed standard screening)*

### 2. Risk Drivers (SHAP Analysis)
Why does the model reject an applicant? I used SHAP values to explain the decisions.

<img width="1002" height="624" alt="Screenshot 2026-02-05 171327" src="https://github.com/user-attachments/assets/dcea5329-aee2-4f7c-b376-297b6ae84e4d" />
*(Figure 2: Key Risk Factors. Notice how **Low CIBIL Scores (Blue dots)** and **Low UPI Usage** push applicants towards the "High Risk" (Right side) zone, validating the importance of digital footprint in credit scoring.)*

---

## 🧠 Strategic "Risk-Based Pricing" Policy
Instead of a binary "Approve/Reject," I used the model's probability scores to design a tiered pricing strategy:

| Risk Tier | Model Probability | Action | Interest Rate |
| :--- | :--- | :--- | :--- |
| **Tier A (Safe)** | < 30% | Auto-Approve | **11.5%** |
| **Tier B (Medium)** | 30% - 70% | Manual Review | **15.5%** |
| **Tier C (High)** | > 70% | Reject | N/A |

---

## 🛠️ Tech Stack
* **Python:** Data generation and pipeline.
* **SMOTE (Imbalanced-Learn):** Synthetically generating minority class data to fix bias.
* **XGBoost:** Gradient boosting for high-performance classification.
* **SHAP:** For explaining *why* a specific applicant was rejected (Regulatory requirement).

## 🚀 How to Run
1.  **Clone the Repo:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/Fintech-Credit-Risk.git](https://github.com/YOUR_USERNAME/Fintech-Credit-Risk.git)
    ```
2.  **Install Requirements:**
    ```bash
    pip install pandas sklearn xgboost imbalanced-learn shap matplotlib seaborn
    ```
3.  **Step 1: Generate Data**
    ```bash
    python generate_credit_data.py
    ```
4.  **Step 2: Train Model**
    ```bash
    python train_risk_model.py
    ```
