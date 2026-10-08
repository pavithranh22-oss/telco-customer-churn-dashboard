Telco Customer Churn Intelligence Dashboard
A business-intelligence project that turns raw telecom customer data into churn insights, risk scores and recommended actions.
Problem statement
Telecom companies lose revenue when customers leave. This project identifies who churns, why, who is at risk next, and what management should do.
Dataset
Telco Customer Churn (IBM sample data), 7,043 customers, 21 columns
Source: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
File used: WA_Fn-UseC_-Telco-Customer-Churn.csv
Approach
Cleaning: fix blank TotalCharges, remove duplicate IDs, create tenure groups and a churn flag.
KPIs: customers, churn rate, average monthly charge, monthly revenue lost to churn.
Drivers: churn rate by contract, internet service, payment method, tech support, tenure, etc.
Model: Logistic Regression (scikit-learn pipeline, one-hot encoding, scaling). 80/20 stratified split. No target leakage: customer ID and churn-derived columns are excluded.
Risk scoring: out-of-fold churn probabilities, tiered Low (<40%), Medium (40-70%), High (>=70%).
Decision page: Risk, Opportunity, Action.
Dashboard pages
Executive Overview
Churn Drivers
Customer Risk & Model
Risk • Opportunity • Action