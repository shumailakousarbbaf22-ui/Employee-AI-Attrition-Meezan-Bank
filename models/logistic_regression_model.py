# =============================================================================
# Employee Attrition Prediction — Meezan Bank Pakistan
# Model 1: Logistic Regression (Baseline Model)
# Authors: Shumaila Kousar & Siffwah Mumtaz | BBA Semester VIII
# Subject: AI in Business
# =============================================================================
#
# PURPOSE:
#   Logistic Regression serves as the baseline model for this binary
#   classification task (employee leaves = 1, stays = 0). It is fast,
#   interpretable, and establishes a performance benchmark that the
#   more complex models (Random Forest, Gradient Boosting) must beat.
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# STEP 1 — LOAD DATA
# =============================================================================

# Replace with your actual dataset path
df = pd.read_csv("employee_attrition_data.csv")

print("=" * 60)
print("  MEEZAN BANK — EMPLOYEE ATTRITION PREDICTION")
print("  Model: Logistic Regression (Baseline)")
print("=" * 60)
print(f"\nDataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}\n")

# =============================================================================
# STEP 2 — DATA PREPROCESSING
# =============================================================================

print(">> Step 2: Preprocessing...\n")

# 2a. Drop duplicates
df.drop_duplicates(inplace=True)
print(f"  Rows after removing duplicates: {len(df)}")

# 2b. Drop irrelevant / near-zero variance columns
cols_to_drop = [col for col in ['EmployeeNumber', 'EmployeeCount', 'Over18', 'StandardHours']
                if col in df.columns]
df.drop(columns=cols_to_drop, inplace=True)

# 2c. Encode target variable: Yes → 1, No → 0
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# 2d. Impute missing values
for col in df.select_dtypes(include='number').columns:
    df[col].fillna(df[col].median(), inplace=True)

for col in df.select_dtypes(include='object').columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

# 2e. Encode OverTime binary column
if 'OverTime' in df.columns:
    df['OverTime'] = df['OverTime'].map({'Yes': 1, 'No': 0})

# 2f. One-hot encode remaining categorical variables
df = pd.get_dummies(df, drop_first=True)

print(f"  Final feature count after encoding: {df.shape[1] - 1}")

# =============================================================================
# STEP 3 — TRAIN / TEST SPLIT & SCALING
# =============================================================================

X = df.drop('Attrition', axis=1)
y = df['Attrition']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Standard scaling — essential for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"\n  Training samples : {X_train_scaled.shape[0]}")
print(f"  Test samples     : {X_test_scaled.shape[0]}")

# =============================================================================
# STEP 4 — TRAIN LOGISTIC REGRESSION MODEL
# =============================================================================

print("\n>> Step 4: Training Logistic Regression model...\n")

lr_model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',   # handles class imbalance (84% stayed, 16% left)
    solver='lbfgs',
    random_state=42
)
lr_model.fit(X_train_scaled, y_train)

# =============================================================================
# STEP 5 — PREDICTIONS & EVALUATION METRICS
# =============================================================================

y_pred       = lr_model.predict(X_test_scaled)
y_pred_proba = lr_model.predict_proba(X_test_scaled)[:, 1]

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall    = recall_score(y_test, y_pred, zero_division=0)
f1        = f1_score(y_test, y_pred, zero_division=0)

print("=" * 60)
print("  LOGISTIC REGRESSION — EVALUATION RESULTS")
print("=" * 60)
print(f"  Accuracy  : {accuracy:.4f}  ({accuracy*100:.2f}%)")
print(f"  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}  ← critical for catching at-risk employees")
print(f"  F1 Score  : {f1:.4f}  ← best metric for imbalanced classes")
print()
print("  Full Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Stayed', 'Left']))

# Cross-validation (5-fold)
cv_scores = cross_val_score(lr_model, X_train_scaled, y_train, cv=5, scoring='f1')
print(f"  5-Fold Cross-Validation F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# =============================================================================
# STEP 6 — VISUALISATIONS
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Logistic Regression — Model Evaluation\nMeezan Bank Employee Attrition",
             fontsize=14, fontweight='bold', y=1.02)

# --- 6a. Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Predicted: Stayed', 'Predicted: Left'],
            yticklabels=['Actual: Stayed', 'Actual: Left'])
axes[0].set_title('Confusion Matrix', fontweight='bold')
axes[0].set_ylabel('Actual Label')
axes[0].set_xlabel('Predicted Label')

# Annotate FN cost note
tn, fp, fn, tp = cm.ravel()
axes[0].text(0.5, -0.18,
    f"TN={tn}  FP={fp}  FN={fn}  TP={tp}\n"
    f"⚠ False Negatives ({fn}) = at-risk employees missed",
    ha='center', va='center', transform=axes[0].transAxes,
    fontsize=8, color='darkred')

# --- 6b. ROC Curve ---
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
axes[1].plot(fpr, tpr, color='steelblue', lw=2,
             label=f'ROC Curve (AUC = {roc_auc:.3f})')
axes[1].plot([0, 1], [0, 1], color='grey', linestyle='--', lw=1, label='Random Classifier')
axes[1].set_xlim([0.0, 1.0])
axes[1].set_ylim([0.0, 1.05])
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curve', fontweight='bold')
axes[1].legend(loc='lower right')
axes[1].grid(alpha=0.3)

# --- 6c. Metrics Bar Chart ---
metrics      = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
metric_vals  = [accuracy, precision, recall, f1]
colors       = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']
bars = axes[2].bar(metrics, metric_vals, color=colors, width=0.5, edgecolor='white')
axes[2].set_ylim(0, 1.1)
axes[2].set_title('Performance Metrics', fontweight='bold')
axes[2].set_ylabel('Score')
for bar, val in zip(bars, metric_vals):
    axes[2].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.02,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
axes[2].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("logistic_regression_results.png", dpi=150, bbox_inches='tight')
plt.show()
print("\n  Chart saved → logistic_regression_results.png")

# =============================================================================
# STEP 7 — TOP COEFFICIENTS (Feature Influence)
# =============================================================================

coef_df = pd.DataFrame({
    'Feature'    : X.columns,
    'Coefficient': lr_model.coef_[0]
}).sort_values('Coefficient', ascending=False)

top_positive = coef_df.head(10)    # factors that increase attrition risk
top_negative = coef_df.tail(10)    # factors that reduce attrition risk
top_features = pd.concat([top_positive, top_negative])

plt.figure(figsize=(10, 7))
colors = ['#C44E52' if c > 0 else '#55A868' for c in top_features['Coefficient']]
plt.barh(top_features['Feature'], top_features['Coefficient'], color=colors, edgecolor='white')
plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
plt.title('Logistic Regression — Top Feature Coefficients\n'
          'Red = Increases Attrition Risk | Green = Reduces Risk',
          fontsize=12, fontweight='bold')
plt.xlabel('Coefficient Value')
plt.tight_layout()
plt.savefig("logistic_regression_coefficients.png", dpi=150, bbox_inches='tight')
plt.show()
print("  Chart saved → logistic_regression_coefficients.png\n")

# =============================================================================
# SUMMARY
# =============================================================================

print("=" * 60)
print("  LOGISTIC REGRESSION — SUMMARY")
print("=" * 60)
print(f"  This baseline model achieves an F1 score of {f1:.4f}.")
print(f"  AUC-ROC: {roc_auc:.4f}")
print(f"  It correctly flags {tp} of {tp + fn} actual leavers (Recall = {recall:.2%}).")
print(f"  False Negatives = {fn} — employees who will leave but were not flagged.")
print("\n  ➜ Compare this against Random Forest and Gradient Boosting")
print("    to determine the best model for HR deployment.\n")
