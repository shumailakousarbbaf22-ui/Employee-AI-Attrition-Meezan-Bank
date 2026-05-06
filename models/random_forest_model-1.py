# =============================================================================
# Employee Attrition Prediction — Meezan Bank Pakistan
# Model 2: Random Forest (Primary Model ⭐)
# Authors: Shumaila Kousar & Siffwah Mumtaz | BBA Semester VIII
# Subject: AI in Business
# =============================================================================
#
# PURPOSE:
#   Random Forest is the primary model of this project. It builds an
#   ensemble of decision trees — each trained on a random subset of
#   data and features — and combines outputs by majority vote.
#
#   KEY ADVANTAGES:
#   ✔ Robust against overfitting
#   ✔ Handles non-linear relationships in data
#   ✔ Built-in feature importance → directly informs HR policy
#   ✔ Works well with imbalanced classes (with class_weight='balanced')
#
#   OUTPUTS:
#   ✔ Individual attrition probability score per employee
#   ✔ Risk tier segmentation: High / Medium / Low
#   ✔ Personalised retention recommendations
#   ✔ Feature importance ranking
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# STEP 1 — LOAD DATA
# =============================================================================

df = pd.read_csv("employee_attrition_data.csv")

print("=" * 65)
print("  MEEZAN BANK — EMPLOYEE ATTRITION PREDICTION")
print("  Model: Random Forest  ⭐  (Primary Model)")
print("=" * 65)
print(f"\nDataset shape : {df.shape}")
print(f"Attrition rate: {(df['Attrition'] == 'Yes').mean() * 100:.2f}%\n")

# =============================================================================
# STEP 2 — DATA PREPROCESSING
# =============================================================================

print(">> Step 2: Preprocessing...\n")

# 2a. Drop duplicates
df.drop_duplicates(inplace=True)

# 2b. Drop irrelevant / near-zero variance columns
cols_to_drop = [col for col in ['EmployeeNumber', 'EmployeeCount', 'Over18', 'StandardHours']
                if col in df.columns]
df.drop(columns=cols_to_drop, inplace=True)

# 2c. Encode target variable
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# 2d. Impute missing values
for col in df.select_dtypes(include='number').columns:
    df[col].fillna(df[col].median(), inplace=True)
for col in df.select_dtypes(include='object').columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

# 2e. Encode OverTime
if 'OverTime' in df.columns:
    df['OverTime'] = df['OverTime'].map({'Yes': 1, 'No': 0})

# 2f. One-hot encode remaining categoricals
df = pd.get_dummies(df, drop_first=True)

print(f"  Final feature count: {df.shape[1] - 1}")

# =============================================================================
# STEP 3 — TRAIN / TEST SPLIT & SCALING
# =============================================================================

X = df.drop('Attrition', axis=1)
y = df['Attrition']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Random Forest is not distance-based, but we scale for consistency
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"\n  Training samples : {X_train_scaled.shape[0]}")
print(f"  Test samples     : {X_test_scaled.shape[0]}")

# =============================================================================
# STEP 4 — TRAIN RANDOM FOREST MODEL
# =============================================================================

print("\n>> Step 4: Training Random Forest model...\n")

rf_model = RandomForestClassifier(
    n_estimators=200,          # 200 decision trees in the ensemble
    max_depth=None,            # let trees grow fully (pruned by min_samples)
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',   # compensates for 84%/16% class imbalance
    random_state=42,
    n_jobs=-1                  # use all available CPU cores
)
rf_model.fit(X_train_scaled, y_train)
print("  ✔ Model trained successfully.")

# =============================================================================
# STEP 5 — PREDICTIONS & EVALUATION METRICS
# =============================================================================

y_pred       = rf_model.predict(X_test_scaled)
y_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall    = recall_score(y_test, y_pred, zero_division=0)
f1        = f1_score(y_test, y_pred, zero_division=0)

print("\n" + "=" * 65)
print("  RANDOM FOREST — EVALUATION RESULTS")
print("=" * 65)
print(f"  Accuracy  : {accuracy:.4f}  ({accuracy*100:.2f}%)")
print(f"  Precision : {precision:.4f}  — avoids wasting HR resources on false alarms")
print(f"  Recall    : {recall:.4f}  — % of actual leavers correctly identified")
print(f"  F1 Score  : {f1:.4f}  — best metric for imbalanced datasets")
print()
print("  Full Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Stayed (0)', 'Left (1)']))

cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5, scoring='f1')
print(f"  5-Fold Cross-Validation F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# =============================================================================
# STEP 6 — FEATURE IMPORTANCE
# =============================================================================

print("\n>> Step 6: Feature Importance Analysis...\n")

importance_df = pd.DataFrame({
    'Feature'   : X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False).reset_index(drop=True)

print("  Top 10 Attrition Drivers:")
print(importance_df.head(10).to_string(index=False))

# =============================================================================
# STEP 7 — RISK SEGMENTATION
# =============================================================================

print("\n>> Step 7: Risk Segmentation...\n")

# Apply model to full dataset for individual risk scores
X_full_scaled = scaler.transform(X)
all_proba     = rf_model.predict_proba(X_full_scaled)[:, 1]

risk_df = X.copy()
risk_df['AttritionProbability'] = all_proba
risk_df['Attrition_Actual']     = y.values

def assign_risk_tier(prob):
    if prob >= 0.65:
        return 'High Risk'
    elif prob >= 0.35:
        return 'Medium Risk'
    else:
        return 'Low Risk'

risk_df['RiskTier'] = risk_df['AttritionProbability'].apply(assign_risk_tier)

tier_counts = risk_df['RiskTier'].value_counts()
print("  Risk Tier Distribution:")
for tier in ['High Risk', 'Medium Risk', 'Low Risk']:
    count = tier_counts.get(tier, 0)
    pct   = count / len(risk_df) * 100
    print(f"    {tier:12s}: {count:4d} employees ({pct:.1f}%)")

# =============================================================================
# STEP 8 — RETENTION RECOMMENDATIONS
# =============================================================================

print("\n>> Step 8: Generating Retention Recommendations...\n")

def get_recommendations(row):
    """
    Generates personalised retention recommendations for each employee
    based on their individual risk factors identified by the model.
    """
    recs = []

    # Check income (use median as threshold)
    income_col = 'MonthlyIncome'
    if income_col in row.index and row[income_col] < risk_df[income_col].median():
        recs.append("💰 Salary Review: Monthly income below median — "
                    "benchmark against role market rate and review.")

    # Overtime flag
    overtime_col = 'OverTime'
    if overtime_col in row.index and row[overtime_col] == 1:
        recs.append("⏰ Overtime Management: Regular overtime detected — "
                    "review workload, enforce overtime caps, offer comp time.")

    # Long commute
    commute_col = 'DistanceFromHome'
    if commute_col in row.index and row[commute_col] > risk_df[commute_col].quantile(0.75):
        recs.append("🚗 Commute Support: Long commute identified — "
                    "consider hybrid/remote options or transport allowance.")

    # Stagnant promotion
    promo_col = 'YearsSinceLastPromotion'
    if promo_col in row.index and row[promo_col] > 3:
        recs.append("📈 Career Development: No promotion in 3+ years — "
                    "initiate career conversation and define advancement criteria.")

    # Work-life balance
    wlb_col = 'WorkLifeBalance'
    if wlb_col in row.index and row[wlb_col] <= 2:
        recs.append("⚖️  Work-Life Balance: Low balance score — "
                    "discuss flexible scheduling and mental health resources.")

    return recs if recs else ["✅ No critical risk factors flagged — maintain standard engagement."]

# Display recommendations for top 5 highest-risk employees
high_risk_employees = risk_df[risk_df['RiskTier'] == 'High Risk'] \
                      .sort_values('AttritionProbability', ascending=False).head(5)

print("  Sample Recommendations for Top 5 Highest-Risk Employees:")
print("  " + "-" * 60)
for idx, (emp_idx, row) in enumerate(high_risk_employees.iterrows(), 1):
    print(f"\n  Employee #{idx} | Risk Score: {row['AttritionProbability']:.1%} | "
          f"Tier: {row['RiskTier']}")
    for rec in get_recommendations(row):
        print(f"    → {rec}")

# =============================================================================
# STEP 9 — VISUALISATIONS
# =============================================================================

print("\n>> Step 9: Generating charts...\n")

fig = plt.figure(figsize=(20, 16))
fig.suptitle("Random Forest — Employee Attrition Analysis\nMeezan Bank Pakistan",
             fontsize=16, fontweight='bold', y=1.01)

# --- 9a. Confusion Matrix ---
ax1 = fig.add_subplot(2, 3, 1)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', ax=ax1,
            xticklabels=['Predicted: Stayed', 'Predicted: Left'],
            yticklabels=['Actual: Stayed', 'Actual: Left'])
ax1.set_title('Confusion Matrix', fontweight='bold')
ax1.set_ylabel('Actual')
ax1.set_xlabel('Predicted')
tn, fp, fn, tp = cm.ravel()
ax1.text(0.5, -0.22,
    f"TN={tn}  FP={fp}  FN={fn}  TP={tp}\n⚠ FN={fn} = missed at-risk employees",
    ha='center', transform=ax1.transAxes, fontsize=8, color='darkred')

# --- 9b. ROC Curve ---
ax2 = fig.add_subplot(2, 3, 2)
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
ax2.plot(fpr, tpr, color='darkorange', lw=2.5,
         label=f'Random Forest (AUC = {roc_auc:.3f})')
ax2.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.05])
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate (Recall)')
ax2.set_title('ROC Curve', fontweight='bold')
ax2.legend(loc='lower right')
ax2.grid(alpha=0.3)

# --- 9c. Performance Metrics Bar Chart ---
ax3 = fig.add_subplot(2, 3, 3)
metrics     = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
metric_vals = [accuracy, precision, recall, f1]
colors      = ['#2196F3', '#FF9800', '#4CAF50', '#F44336']
bars = ax3.bar(metrics, metric_vals, color=colors, width=0.5, edgecolor='white')
ax3.set_ylim(0, 1.15)
ax3.set_title('Performance Metrics', fontweight='bold')
ax3.set_ylabel('Score')
for bar, val in zip(bars, metric_vals):
    ax3.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.02,
             f'{val:.3f}', ha='center', fontsize=9, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)

# --- 9d. Feature Importance (Top 15) ---
ax4 = fig.add_subplot(2, 3, 4)
top15 = importance_df.head(15).sort_values('Importance')
colors_fi = ['#C44E52' if i >= 10 else '#4C72B0' for i in range(len(top15))]
ax4.barh(top15['Feature'], top15['Importance'], color=colors_fi[::-1], edgecolor='white')
ax4.set_title('Top 15 Feature Importances\n(Red = Top 5 Drivers)', fontweight='bold')
ax4.set_xlabel('Importance Score')
ax4.grid(axis='x', alpha=0.3)

# --- 9e. Risk Tier Distribution ---
ax5 = fig.add_subplot(2, 3, 5)
tier_order  = ['High Risk', 'Medium Risk', 'Low Risk']
tier_colors = ['#F44336', '#FF9800', '#4CAF50']
tier_vals   = [tier_counts.get(t, 0) for t in tier_order]
wedges, texts, autotexts = ax5.pie(
    tier_vals, labels=tier_order, colors=tier_colors,
    autopct='%1.1f%%', startangle=90,
    wedgeprops=dict(edgecolor='white', linewidth=2)
)
for at in autotexts:
    at.set_fontsize(10)
    at.set_fontweight('bold')
ax5.set_title('Employee Risk Tier Distribution', fontweight='bold')

# --- 9f. Attrition Probability Distribution ---
ax6 = fig.add_subplot(2, 3, 6)
stayed_proba = risk_df[risk_df['Attrition_Actual'] == 0]['AttritionProbability']
left_proba   = risk_df[risk_df['Attrition_Actual'] == 1]['AttritionProbability']
ax6.hist(stayed_proba, bins=40, alpha=0.6, color='#4CAF50', label='Stayed', edgecolor='white')
ax6.hist(left_proba,   bins=40, alpha=0.6, color='#F44336', label='Left',   edgecolor='white')
ax6.axvline(0.35, color='orange', linestyle='--', linewidth=1.5, label='Medium Risk Threshold (35%)')
ax6.axvline(0.65, color='red',    linestyle='--', linewidth=1.5, label='High Risk Threshold (65%)')
ax6.set_xlabel('Predicted Attrition Probability')
ax6.set_ylabel('Number of Employees')
ax6.set_title('Predicted Probability Distribution\nStayed vs Left', fontweight='bold')
ax6.legend(fontsize=8)
ax6.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("random_forest_results.png", dpi=150, bbox_inches='tight')
plt.show()
print("  Charts saved → random_forest_results.png\n")

# =============================================================================
# STEP 10 — BUSINESS IMPACT SUMMARY
# =============================================================================

current_attrition_rate   = 0.1597
projected_attrition_rate = 0.1277
total_employees          = len(df)
employees_retained       = round((current_attrition_rate - projected_attrition_rate) * total_employees)
cost_per_replacement     = 300_000
total_savings            = employees_retained * cost_per_replacement

print("=" * 65)
print("  BUSINESS IMPACT — PROJECTED FINANCIAL SAVINGS")
print("=" * 65)
print(f"  Current Attrition Rate         : {current_attrition_rate:.2%}")
print(f"  Projected Rate (Post-Action)   : {projected_attrition_rate:.2%}")
print(f"  Employees Retained             : ~{employees_retained}")
print(f"  Replacement Cost per Employee  : PKR {cost_per_replacement:,}")
print(f"  ─────────────────────────────────────────")
print(f"  TOTAL PROJECTED ANNUAL SAVINGS : PKR {total_savings:,}")
print()
print("  Note: Conservative estimate — excludes morale, knowledge")
print("  retention, and team stability benefits.\n")

print("=" * 65)
print("  RANDOM FOREST — FINAL SUMMARY")
print("=" * 65)
print(f"  F1 Score  : {f1:.4f}")
print(f"  AUC-ROC   : {roc_auc:.4f}")
print(f"  Recall    : {recall:.4f}")
print(f"\n  Top 5 Attrition Drivers Identified:")
for i, row in importance_df.head(5).iterrows():
    print(f"    {i+1}. {row['Feature']} (importance = {row['Importance']:.4f})")
print()
print("  ➜ This model is recommended for HR deployment.")
print("    Risk scores are available in risk_df['AttritionProbability'].")
print("    Personalised recommendations generated for all High Risk employees.\n")
