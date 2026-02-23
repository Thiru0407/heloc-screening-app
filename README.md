# HELOC Credit Risk Screening Tool

## Overview
This project develops a Machine Learning-based Decision Support System (DSS) for preliminary screening of Home Equity Line of Credit (HELOC) applications.

The system predicts whether an applicant is likely to be low-risk (Good) or high-risk (Bad) and provides interpretable explanations for decisions.

## Model Information
- Model Type: Logistic Regression
- Accuracy: 71.85%
- ROC-AUC: 0.79
- Features Used:
  - ExternalRiskEstimate
  - NumInqLast6M
  - NetFractionRevolvingBurden
  - NumSatisfactoryTrades
  - AverageMInFile

## Deployment
The application is deployed using Streamlit Community Cloud.

## How It Works
1. User enters applicant credit information.
2. The model predicts approval probability.
3. If risk is high, the system provides:
   - Main risk reasons
   - Suggested improvement steps

## Files Included
- app.py → Streamlit web application
- heloc_model.pkl → Trained Logistic Regression model
- requirements.txt → Required Python packages

## Disclaimer
This tool is for educational and preliminary screening purposes only. Final lending decisions should involve professional credit evaluation.
