import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# LOAD DATA
df = pd.read_csv("brain_tumor_dataset.csv")
print("Dataset loaded successfully!")

# Show dataset info
print("DATASET INFO:")
print(df.info())

# Show summary statistics
print("SUMMARY STATISTICS:")
print(df.describe())

#Deop unnecessary columns (Patient ID)
df.drop('Patient_ID', axis = 1, inplace = True)


histology = df['Histology'].unique()
print("Unique Histology values:", histology)

unique_locations = df['Location'].unique()
print("Unique Location values:", unique_locations)

print("VALUE COUNTS (TARGET):")
print(df['Tumor_Type'].value_counts())

# DEFINE FEATURES (X) AND TARGET (y)
if 'Patient_ID' in df.columns:
    df = df.drop(columns=['Patient_ID'])

# Handle missing values in target variable
df.dropna(subset=['Tumor_Type'], inplace=True)

# survival rate outcome

SURVIVAL_THRESHOLD = 80
df['Survival_Outcome']= (df['Survival_Rate'] > SURVIVAL_THRESHOLD).astype(int)


X = df.drop(columns=['Tumor_Type'])
y = df['Tumor_Type']

# ONE-HOT ENCODING FOR CATEGORICAL FEATURES
X = pd.get_dummies(X, drop_first=True)

# TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# PRE-PROCESSING (SCALING) - Only for SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n" + "="*60)
print("TRAINING AND EVALUATING MODELS")
print("="*60)

# ============================
# 1. SVM MODEL
# ============================
print("\n" + "-"*40)
print("SVM MODEL (RBF Kernel)")
print("-"*40)

svm_model = SVC(kernel='rbf',C=1.0, class_weight='balanced', random_state=42)
svm_model.fit(X_train_scaled, y_train)
y_pred_svm = svm_model.predict(X_test_scaled)

# SVM Metrics
accuracy_svm = accuracy_score(y_test, y_pred_svm)
precision_svm = precision_score(y_test, y_pred_svm, average='weighted', zero_division=0)
recall_svm = recall_score(y_test, y_pred_svm, average='weighted', zero_division=0)
f1_svm = f1_score(y_test, y_pred_svm, average='weighted', zero_division=0)

print(f"Accuracy:  {accuracy_svm:.4f}")
print(f"Precision: {precision_svm:.4f}")
print(f"Recall:    {recall_svm:.4f}")
print(f"F1-Score:  {f1_svm:.4f}")

# SVM Confusion Matrix
cm_svm = confusion_matrix(y_test, y_pred_svm)

# ============================
# 2. RANDOM FOREST MODEL
# ============================
print("\n" + "-"*40)
print("RANDOM FOREST MODEL")
print("-"*40)

# Random Forest doesn't require scaling, so we use the original features
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Random Forest Metrics
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf, average='weighted', zero_division=0)
recall_rf = recall_score(y_test, y_pred_rf, average='weighted', zero_division=0)
f1_rf = f1_score(y_test, y_pred_rf, average='weighted', zero_division=0)

print(f"Accuracy:  {accuracy_rf:.4f}")
print(f"Precision: {precision_rf:.4f}")
print(f"Recall:    {recall_rf:.4f}")
print(f"F1-Score:  {f1_rf:.4f}")

# Random Forest Confusion Matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)

# ============================
# MODEL COMPARISON SUMMARY
# ============================
print("\n" + "="*60)
print("MODEL COMPARISON SUMMARY")
print("="*60)

comparison_df = pd.DataFrame({
    'Model': ['SVM', 'Random Forest'],
    'Accuracy': [accuracy_svm, accuracy_rf],
    'Precision': [precision_svm, precision_rf],
    'Recall': [recall_svm, recall_rf],
    'F1-Score': [f1_svm, f1_rf]
})

print(comparison_df.to_string(index=False))

# ============================
# CONFUSION MATRICES VISUALIZATION
# ============================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Get class labels for annotations
class_labels = sorted(y.unique())

# Plot SVM Confusion Matrix
im1 = axes[0].imshow(cm_svm, interpolation='nearest', cmap='Blues')
axes[0].set_title("SVM - Confusion Matrix", fontsize=14, fontweight='bold')
axes[0].set_xlabel("Predicted Label", fontsize=12)
axes[0].set_ylabel("True Label", fontsize=12)
axes[0].set_xticks(np.arange(len(class_labels)))
axes[0].set_yticks(np.arange(len(class_labels)))
axes[0].set_xticklabels(class_labels)
axes[0].set_yticklabels(class_labels)

# Add numbers on SVM confusion matrix cells
for i in range(cm_svm.shape[0]):
    for j in range(cm_svm.shape[1]):
        axes[0].text(j, i, cm_svm[i, j], ha='center', va='center', 
                    fontsize=12, color='white' if cm_svm[i, j] > cm_svm.max()/2 else 'black')

# Plot Random Forest Confusion Matrix
im2 = axes[1].imshow(cm_rf, interpolation='nearest', cmap='Greens')
axes[1].set_title("Random Forest - Confusion Matrix", fontsize=14, fontweight='bold')
axes[1].set_xlabel("Predicted Label", fontsize=12)
axes[1].set_ylabel("True Label", fontsize=12)
axes[1].set_xticks(np.arange(len(class_labels)))
axes[1].set_yticks(np.arange(len(class_labels)))
axes[1].set_xticklabels(class_labels)
axes[1].set_yticklabels(class_labels)

# Add numbers on Random Forest confusion matrix cells
for i in range(cm_rf.shape[0]):
    for j in range(cm_rf.shape[1]):
        axes[1].text(j, i, cm_rf[i, j], ha='center', va='center', 
                    fontsize=12, color='white' if cm_rf[i, j] > cm_rf.max()/2 else 'black')

# Add colorbars
plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()

# ============================
# DETAILED CLASSIFICATION REPORTS
# ============================
print("\n" + "="*60)
print("DETAILED CLASSIFICATION REPORTS")
print("="*60)

print("\nSVM CLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred_svm, zero_division=0))

print("\nRANDOM FOREST CLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred_rf, zero_division=0))

# ============================
# FEATURE IMPORTANCE (Random Forest only)
# ============================
print("\n" + "="*60)
print("RANDOM FOREST FEATURE IMPORTANCE (Top 10)")
print("="*60)

# Get feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
})

# Sort by importance and get top 10
top_features = feature_importance.sort_values('importance', ascending=False).head(10)

print("\nTop 10 Most Important Features:")
for idx, row in top_features.iterrows():
    print(f"{row['feature']}: {row['importance']:.4f}")

# Optional: Feature importance visualization
fig2, ax = plt.subplots(figsize=(10, 6))
top_features.sort_values('importance').plot(kind='barh', x='feature', y='importance', ax=ax, color='green')
ax.set_title('Top 10 Feature Importances - Random Forest', fontsize=14, fontweight='bold')
ax.set_xlabel('Importance Score', fontsize=12)
plt.tight_layout()
plt.show()

