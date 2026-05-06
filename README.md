🏦 Employee Attrition Prediction & Retention Strategy
Meezan Bank Pakistan — AI in Business | BBA Semester VIII
Predicting employee flight risk before it's too late — using machine learning to save talent and cut costs.
📋 Table of Contents
Overview
Problem Statement
Dataset
Project Workflow
Exploratory Data Analysis
Machine Learning Models
Model Performance
Feature Importance
Risk Segmentation
Retention Recommendations
Business Impact
Future Enhancements
Authors
Overview
Employee turnover is one of the most costly and operationally disruptive challenges organisations face. For a financial institution like Meezan Bank Pakistan — where client trust, specialised training, and service continuity are critical — losing experienced staff carries an especially high price.
This project applies machine learning to Human Resource Management, shifting HR from a reactive to a proactive stance. Instead of responding to resignations after they happen, the system predicts which employees are most likely to leave — and why — so targeted interventions can be made before it's too late.
Key outcomes:
Three ML models trained and compared on 3,000 employee records
Individual risk scores assigned to every employee
Top attrition drivers identified and ranked
Projected retention of 96 employees and PKR 28.8 million in annual savings
Problem Statement
Meezan Bank faces a persistent challenge: talented employees are leaving, and by the time a resignation is submitted, the window for meaningful retention has already closed.
This project addresses four specific objectives:
Predict which employees are at highest risk of leaving in the near term
Identify root causes driving attrition across departments and roles
Generate actionable, personalised retention recommendations for at-risk employees
Quantify the financial benefit of reducing attrition through proactive intervention
Dataset
Property
Detail
Records
3,000 employee entries
Features
43 variables
Target Variable
Attrition (Yes = Left / No = Stayed)
Overall Attrition Rate
15.97%
Key Variables
Variable
Description
Age
Employee age in years
MonthlyIncome
Gross monthly salary (PKR)
Department
Organisational department
JobRole
Specific role within department
OverTime
Whether employee regularly works overtime (Yes/No)
DistanceFromHome
Commuting distance in kilometres
WorkLifeBalance
Self-reported balance score (1–4 scale)
YearsSinceLastPromotion
Time elapsed since most recent promotion
Attrition
Target — whether the employee left
⚠️ Class Imbalance Note: The dataset contains significantly more "stayed" records than "left" records (~84% vs ~16%). Evaluation metrics were chosen accordingly — raw accuracy alone is misleading here.
Project Workflow
Code
Exploratory Data Analysis
EDA was the foundation of the entire project — understanding data shape, statistical properties, and the patterns hiding within it before any modelling began.
Attrition Rate Distribution
Code
Attrition by Overtime Status
Code
Attrition by Work-Life Balance Score
Code
📊 Full interactive charts are generated in the Jupyter Notebook. Run the EDA section to view histograms, box plots, correlation heatmaps, and department-level breakdowns.
Machine Learning Models
Three models were developed to address the binary classification task (leave vs. stay):
1. Logistic Regression — Baseline
The natural starting point for binary classification. Fast to train, easy to interpret, and used here to set a performance benchmark for more complex models.
2. Random Forest — Primary Model ⭐
The core model of this project. Builds a large ensemble of decision trees, each trained on a random data and feature subset, then combines outputs via majority vote.
Why Random Forest?
Robust against overfitting
Captures complex, non-linear relationships
Provides built-in feature importance scores — directly translating to HR policy priorities
3. Gradient Boosting — Advanced Comparison
Trees are built sequentially, with each new tree correcting errors from the previous one. Often achieves high accuracy, but requires more careful tuning and is less interpretable.
Model Performance
Because of class imbalance, accuracy alone is insufficient. The following balanced metrics were used:
Metric
What It Measures
Why It Matters
Accuracy
Overall share of correct predictions
Baseline measure; insufficient alone
Precision
Of predicted leavers, how many actually left
Prevents wasting HR resources on false alarms
Recall
Of actual leavers, how many were correctly flagged
Critical — missing a high-risk employee means losing them
F1 Score
Harmonic mean of Precision and Recall
Best single metric for imbalanced datasets
Confusion Matrix — Random Forest
Code
⚠️ False Negatives carry the highest operational cost — an at-risk employee incorrectly classified as "safe" is one the bank loses without warning. Minimising False Negatives was a key priority in model selection and threshold calibration.
📊 Full confusion matrices, ROC curves, and metric comparisons across all three models are visualised in the notebook.
Feature Importance
One of the most valuable outputs of the Random Forest model is its feature importance ranking — a quantified view of which factors most strongly drive the decision to leave.
Top 5 Attrition Drivers
Code
#
Driver
Business Implication
1
Low Monthly Income
Salary is the single strongest predictor of departure
2
Overtime
Regular overtime significantly elevates flight risk at all levels
3
Distance From Home
Long commutes compound stress and erode balance
4
Years Since Last Promotion
Career stagnation drives gradual disengagement
5
Work–Life Balance Score
Low scores are a reliable leading indicator of resignation
📊 A horizontal bar chart of all feature importances is generated in the notebook's Feature Importance section.
Risk Segmentation
The trained Random Forest model assigns an individual attrition probability score to every employee. These scores segment the workforce into three actionable tiers:
Code
Risk Tier
Probability
Recommended HR Response
🔴 High Risk
Above 65%
Immediate, personalised engagement; priority salary and workload review
🟡 Medium Risk
35% – 65%
Proactive check-ins; career conversations; flexibility options
🟢 Low Risk
Below 35%
Standard engagement; monitor for behavioural or performance changes
📊 A risk distribution pie/bar chart showing the proportion of employees in each tier is generated in the Prediction section of the notebook.
Retention Recommendations
Prediction without action has limited value. The system translates risk scores into concrete HR interventions tied directly to each employee's flagged risk factors:
Risk Factor
Recommended Intervention
💰 Low Salary
Salary benchmarking and targeted pay reviews, especially for high-risk high-performers
⏰ Overtime
Policies capping compulsory extended hours; compensatory time-off mechanisms
🚗 Long Commute
Commuter allowances, shuttle services, or hybrid/remote working arrangements
📈 Stagnant Career
Clear, transparent promotion pathways; structured advancement criteria
⚖️ Poor Work-Life Balance
Flexible scheduling, mental health resources, manager burnout-recognition training
Business Impact
Every resignation costs far more than filling an empty desk. Recruitment, onboarding, and productivity ramp-up time add up — and in a compliance-sensitive institution, the cost of lost institutional knowledge is even higher.
Projected Financial Savings
Code
These projections are conservative — they account only for direct replacement costs and exclude secondary benefits such as improved morale, team stability, and preserved institutional knowledge.
Future Enhancements
Enhancement
Description
📊 Real-Time HR Dashboard
Surfaces risk scores and trend data for managers without requiring model access
💬 Sentiment Analysis
Applied to surveys and feedback channels to capture signals that structured data misses
🧠 Deep Learning
Neural network architectures to uncover subtle, non-linear patterns in large datasets
🔗 HRIS Integration
Direct connection to existing HR systems so model inputs stay current automatically
🔔 Automated Alerts
Notifies HR partners when an employee's risk score crosses a defined threshold
Authors
Shumaila Kousar & Siffwah Mumtaz
BBA – Semester VIII | AI in Business
Meezan Bank Pakistan Attrition Analysis Project
This project demonstrates that when organisations apply the same analytical rigour to employee data that they apply to financial data, the results can be genuinely transformative — turning reactive HR into a proactive, data-driven retention engine.
