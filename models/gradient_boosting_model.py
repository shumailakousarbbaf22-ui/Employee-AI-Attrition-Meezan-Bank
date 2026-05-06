# =============================================================================
# Employee Attrition Prediction — Meezan Bank Pakistan
# Model 3: Gradient Boosting (Advanced Comparison Model)
# Authors: Shumaila Kousar & Siffwah Mumtaz | BBA Semester VIII
# Subject: AI in Business
# =============================================================================
#
# PURPOSE:
#   Gradient Boosting is included as an advanced comparison model.
#   Unlike Random Forest, which builds trees independently and in parallel,
#   Gradient Boosting builds them SEQUENTIALLY — each new tree specifically
#   targeting and correcting the errors made by the previous one.
#
#   KEY CHARACTERISTICS:
#   ✔ Iterative error-correction → often higher accuracy than Random Forest
#   ✔ Handles complex non-linear patterns effectively
#   ✔ Also provides feature importance scores
#   ✗ Requires more careful hyperparameter tuning
#   ✗ Slower to train than Random Forest
#   ✗ Slightly less interpretable
#
#   THIS FILE ALSO:
#   ✔ Compares all three models side-by-side (LR vs RF vs GB)
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
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
print("  Model: Gradient Boosting (Advanced Comparison)")
print("=" * 65)
print(f"\nDataset shape : {df.shape}")
print(f"Attrition rate: {(df['Attrition'] == 'Yes').mean() * 100:.2f}%\n")

# =============================================================================
# STEP 2 — DATA PREPROCESSING
# =============================================================================

print(">> Step 2: Preprocessing...\n")

df.drop_duplicates(inplace=True)

cols_to_drop = [col for col in ['EmployeeNumber', 'EmployeeCount', 'Over18', 'StandardHours']
                if col in df.columns]
df.drop(columns=cols_to_drop, inplace=True)

df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

for col in df.select_dtypes(include='number').columns:
    df[col].fillna(df[col].median(), inplace=True)
for col in df.select_dtypes(include='object').columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

if 'OverTime' in df.columns:
    df['OverTime'] = df['OverTime'].map({'Yes': 1, 'No': 0})

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

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"\n  Training samples : {X_train_scaled.shape[0]}")
print(f"  Test samples     : {X_test_scaled.shape[0]}")

# =============================================================================
# STEP 4 — TRAIN GRADIENT BOOSTING MODEL
# =============================================================================

print("\n>> Step 4: Training Gradient Boosting model...\n")

gb_model = GradientBoostingClassifier(
    n_estimators=200,       # number of boosting stages (trees)
    learning_rate=0.1,      # shrinks contribution of each tree — controls overfitting
    max_depth=4,            # depth of individual trees
    min_samples_split=5,
    min_samples_leaf=2,
    subsample=0.8,          # fraction of samples used per tree — reduces overfitting
    random_state=42
)
gb_model.fit(X_train_scaled, y_train)
print("  ✔ Model trained successfully.")

# =============================================================================
# STEP 5 — PREDICTIONS & EVALUATION
# =============================================================================

y_pred_gb       = gb_model.predict(X_test_scaled)
y_pred_proba_gb = gb_model.predict_proba(X_test_scaled)[:, 1]

accuracy_gb  = accuracy_score(y_test, y_pred_gb)
precision_gb = precision_score(y_test, y_pred_gb, zero_division=0)
recall_gb    = recall_score(y_test, y_pred_gb, zero_division=0)
f1_gb        = f1_score(y_test, y_pred_gb, zero_division=0)

print("\n" + "=" * 65)
print("  GRADIENT BOOSTING — EVALUATION RESULTS")
print("=" * 65)
print(f"  Accuracy  : {accuracy_gb:.4f}  ({accuracy_gb*100:.2f}%)")
print(f"  Precision : {precision_gb:.4f}")
print(f"  Recall    : {recall_gb:.4f}  ← critical for at-risk detection")
print(f"  F1 Score  : {f1_gb:.4f}")
print()
print("  Full Classification Report:")
print(classification_report(y_test, y_pred_gb, target_names=['Stayed (0)', 'Left (1)']))

cv_scores_gb = cross_val_score(gb_model, X_train_scaled, y_train, cv=5, scoring='f1')
print(f"  5-Fold Cross-Validation F1: {cv_scores_gb.mean():.4f} ± {cv_scores_gb.std():.4f}")

# =============================================================================
# STEP 6 — FEATURE IMPORTANCE
# =============================================================================

print("\n>> Step 6: Feature Importance Analysis...\n")

importance_gb_df = pd.DataFrame({
    'Feature'   : X.columns,
    'Importance': gb_model.feature_importances_
}).sort_values('Importance', ascending=False).reset_index(drop=True)

print("  Top 10 Attrition Drivers (Gradient Boosting):")
print(importance_gb_df.head(10).to_string(index=False))

# =============================================================================
# STEP 7 — TRAIN BASELINE MODELS FOR COMPARISON
# =============================================================================

print("\n>> Step 7: Training baseline models for comparison...\n")

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000, class_weight='balanced',
                               solver='lbfgs', random_state=42)
lr_model.fit(X_train_scaled, y_train)
y_pred_lr       = lr_model.predict(X_test_scaled)
y_pred_proba_lr = lr_model.predict_proba(X_test_scaled)[:, 1]

# Random Forest
rf_model = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                   random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)
y_pred_rf       = rf_model.predict(X_test_scaled)
y_pred_proba_rf = rf_model.predict_proba(X_test_scaled)[:, 1]

# Consolidate metrics
results = {
    'Logistic Regression' : {
        'accuracy' : accuracy_score(y_test, y_pred_lr),
        'precision': precision_score(y_test, y_pred_lr, zero_division=0),
        'recall'   : recall_score(y_test, y_pred_lr, zero_division=0),
        'f1'       : f1_score(y_test, y_pred_lr, zero_division=0),
        'proba'    : y_pred_proba_lr
    },
    'Random Forest' : {
        'accuracy' : accuracy_score(y_test, y_pred_rf),
        'precision': precision_score(y_test, y_pred_rf, zero_division=0),
        'recall'   : recall_score(y_test, y_pred_rf, zero_division=0),
        'f1'       : f1_score(y_test, y_pred_rf, zero_division=0),
        'proba'    : y_pred_proba_rf
    },
    'Gradient Boosting' : {
        'accuracy' : accuracy_gb,
        'precision': precision_gb,
        'recall'   : recall_gb,
        'f1'       : f1_gb,
        'proba'    : y_pred_proba_gb
    }
}

# =============================================================================
# STEP 8 — MODEL COMPARISON TABLE
# =============================================================================

print("\n" + "=" * 65)
print("  MODEL COMPARISON — ALL THREE MODELS")
print("=" * 65)
print(f"  {'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print("  " + "-" * 60)
for name, m in results.items():
    print(f"  {name:<25} {m['accuracy']:>10.4f} {m['precision']:>10.4f} "
          f"{m['recall']:>10.4f} {m['f1']:>10.4f}")

best_model = max(results, key=lambda k: results[k]['f1'])
print(f"\n  ➜ Best model by F1 Score: {best_model}")

# =============================================================================
# STEP 9 — VISUALISATIONS
# =============================================================================

print("\n>> Step 9: Generating charts...\n")

fig = plt.figure(figsize=(20, 16))
fig.suptitle("Gradient Boosting — Employee Attrition Analysis\n"
             "Meezan Bank Pakistan | Including Model Comparison",
             fontsize=16, fontweight='bold', y=1.01)

model_colors = {
    'Logistic Regression': '#4C72B0',
    'Random Forest'      : '#DD8452',
    'Gradient Boosting'  : '#55A868'
}

# --- 9a. Confusion Matrix ---
ax1 = fig.add_subplot(2, 3, 1)
cm_gb = confusion_matrix(y_test, y_pred_gb)
sns.heatmap(cm_gb, annot=True, fmt='d', cmap='Greens', ax=ax1,
            xticklabels=['Pred: Stayed', 'Pred: Left'],
            yticklabels=['Actual: Stayed', 'Actual: Left'])
ax1.set_title('Confusion Matrix\n(Gradient Boosting)', fontweight='bold')
ax1.set_ylabel('Actual')
ax1.set_xlabel('Predicted')
tn, fp, fn, tp = cm_gb.ravel()
ax1.text(0.5, -0.22,
    f"TN={tn}  FP={fp}  FN={fn}  TP={tp}\n⚠ FN={fn} = missed at-risk employees",
    ha='center', transform=ax1.transAxes, fontsize=8, color='darkred')

# --- 9b. ROC Curves — All Three Models ---
ax2 = fig.add_subplot(2, 3, 2)
for name, m in results.items():
    fpr, tpr, _ = roc_curve(y_test, m['proba'])
    roc_auc = auc(fpr, tpr)
    ax2.plot(fpr, tpr, lw=2, color=model_colors[name],
             label=f'{name} (AUC = {roc_auc:.3f})')
ax2.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.05])
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('ROC Curves — All Models', fontweight='bold')
ax2.legend(loc='lower right', fontsize=8)
ax2.grid(alpha=0.3)

# --- 9c. F1 Score Comparison ---
ax3 = fig.add_subplot(2, 3, 3)
model_names = list(results.keys())
f1_scores   = [results[m]['f1'] for m in model_names]
bar_colors  = [model_colors[m] for m in model_names]
bars = ax3.bar(model_names, f1_scores, color=bar_colors, width=0.5, edgecolor='white')
ax3.set_ylim(0, 1.15)
ax3.set_title('F1 Score Comparison\n(All Models)', fontweight='bold')
ax3.set_ylabel('F1 Score')
for bar, val in zip(bars, f1_scores):
    ax3.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.02,
             f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')
ax3.tick_params(axis='x', labelrotation=10)
ax3.grid(axis='y', alpha=0.3)

# --- 9d. Feature Importance (Gradient Boosting, Top 15) ---
ax4 = fig.add_subplot(2, 3, 4)
top15_gb = importance_gb_df.head(15).sort_values('Importance')
colors_fi = ['#C44E52' if i >= 10 else '#55A868' for i in range(len(top15_gb))]
ax4.barh(top15_gb['Feature'], top15_gb['Importance'],
         color=colors_fi[::-1], edgecolor='white')
ax4.set_title('Top 15 Feature Importances\n(Gradient Boosting — Red = Top 5)',
              fontweight='bold')
ax4.set_xlabel('Importance Score')
ax4.grid(axis='x', alpha=0.3)

# --- 9e. Full Metrics Comparison (Grouped Bar) ---
ax5 = fig.add_subplot(2, 3, 5)
metrics_list = ['accuracy', 'precision', 'recall', 'f1']
metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1']
x = np.arange(len(metrics_list))
width = 0.25
for i, (name, m) in enumerate(results.items()):
    vals = [m[metric] for metric in metrics_list]
    bars = ax5.bar(x + i * width, vals, width, label=name,
                   color=model_colors[name], edgecolor='white', alpha=0.85)
ax5.set_xticks(x + width)
ax5.set_xticklabels(metric_labels)
ax5.set_ylim(0, 1.2)
ax5.set_title('All Metrics — Model Comparison', fontweight='bold')
ax5.set_ylabel('Score')
ax5.legend(fontsize=8)
ax5.grid(axis='y', alpha=0.3)

# --- 9f. Attrition Probability Distribution (GB) ---
ax6 = fig.add_subplot(2, 3, 6)
gb_proba_df = pd.DataFrame({
    'Probability': y_pred_proba_gb,
    'Actual'     : y_test.values
})
stayed = gb_proba_df[gb_proba_df['Actual'] == 0]['Probability']
left   = gb_proba_df[gb_proba_df['Actual'] == 1]['Probability']
ax6.hist(stayed, bins=40, alpha=0.6, color='#55A868', label='Stayed', edgecolor='white')
ax6.hist(left,   bins=40, alpha=0.6, color='#C44E52', label='Left',   edgecolor='white')
ax6.axvline(0.35, color='orange', linestyle='--', lw=1.5, label='Medium Risk (35%)')
ax6.axvline(0.65, color='red',    linestyle='--', lw=1.5, label='High Risk (65%)')
ax6.set_xlabel('Predicted Attrition Probability')
ax6.set_ylabel('Number of Employees')
ax6.set_title('Predicted Probability Distribution\n(Gradient Boosting)', fontweight='bold')
ax6.legend(fontsize=8)
ax6.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("gradient_boosting_results.png", dpi=150, bbox_inches='tight')
plt.show()
print("  Charts saved → gradient_boosting_results.png\n")

# =============================================================================
# STEP 10 — FINAL RECOMMENDATION
# =============================================================================

fpr_gb, tpr_gb, _ = roc_curve(y_test, y_pred_proba_gb)
roc_auc_gb = auc(fpr_gb, tpr_gb)

print("=" * 65)
print("  GRADIENT BOOSTING — FINAL SUMMARY")
print("=" * 65)
print(f"  Accuracy  : {accuracy_gb:.4f}")
print(f"  Precision : {precision_gb:.4f}")
print(f"  Recall    : {recall_gb:.4f}")
print(f"  F1 Score  : {f1_gb:.4f}")
print(f"  AUC-ROC   : {roc_auc_gb:.4f}")
print()

print("=" * 65)
print("  FINAL MODEL RECOMMENDATION")
print("=" * 65)
print(f"\n  Best model by F1 Score: ★  {best_model}  ★")
print()
print("  Interpretation Guide:")
print("  ─────────────────────────────────────────────────────")
print("  F1 Score  → Best for imbalanced datasets like this one")
print("  Recall    → Prioritise if catching ALL leavers is critical")
print("  Precision → Prioritise if HR resources are very limited")
print("  AUC-ROC   → Overall discriminative ability of the model")
print()
print("  For Meezan Bank HR deployment, the recommended model is")
print("  the one with the highest Recall × F1 balance — ensuring")
print("  maximum at-risk employees are identified before departure.\n")
