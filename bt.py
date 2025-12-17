import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================
# LOAD AND EXPLORE DATA
# ============================
print("Loading dataset...")
df = pd.read_csv("brain_tumor_dataset.csv")

# Display basic info
print(f"Dataset shape: {df.shape}")
print(f"\nMissing values:\n{df.isnull().sum()}")

# ============================
# DATA PREPROCESSING
# ============================
def preprocess_data(df):
    """Preprocess the brain tumor dataset"""
    df_clean = df.copy()
    
    # Drop Patient_ID
    if 'Patient_ID' in df_clean.columns:
        df_clean = df_clean.drop(columns=['Patient_ID'])
    
    # Convert categorical variables
    categorical_cols = ['Gender', 'Location', 'Histology', 'Symptom_1', 'Symptom_2', 'Symptom_3',
                       'Radiation_Treatment', 'Surgery_Performed', 'Chemotherapy', 
                       'Family_History', 'MRI_Result', 'Follow_Up_Required']
    
    # Encode binary variables
    binary_mapping = {'Yes': 1, 'No': 0, 'Positive': 1, 'Negative': 0}
    for col in ['Radiation_Treatment', 'Surgery_Performed', 'Chemotherapy', 
                'Family_History', 'MRI_Result', 'Follow_Up_Required']:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].map(binary_mapping).fillna(0)
    
    # Encode Gender
    df_clean['Gender'] = df_clean['Gender'].map({'Male': 1, 'Female': 0})
    
    # Encode Stage (I=1, II=2, III=3, IV=4)
    stage_mapping = {'I': 1, 'II': 2, 'III': 3, 'IV': 4}
    df_clean['Stage'] = df_clean['Stage'].map(stage_mapping)
    
    # One-hot encode remaining categorical variables
    categorical_to_encode = ['Location', 'Histology', 'Symptom_1', 'Symptom_2', 'Symptom_3']
    df_clean = pd.get_dummies(df_clean, columns=categorical_to_encode, drop_first=True)
    
    # Define target variables
    # 1. Tumor Type (Main Target)
    tumor_type_target = df_clean['Tumor_Type']
    
    # 2. Survival Outcome (High/Low) - New target
    SURVIVAL_THRESHOLD = 70  # You can adjust this threshold
    survival_outcome = (df_clean['Survival_Rate'] > SURVIVAL_THRESHOLD).astype(int)
    
    # 3. Tumor Activity (Based on growth rate and size)
    df_clean['Tumor_Activity'] = np.where(
        (df_clean['Tumor_Size'] > df_clean['Tumor_Size'].median()) & 
        (df_clean['Tumor_Growth_Rate'] > df_clean['Tumor_Growth_Rate'].median()),
        1, 0  # 1 = High Activity, 0 = Low Activity
    )
    activity_target = df_clean['Tumor_Activity']
    
    # Prepare features (exclude targets and survival rate)
    features_to_drop = ['Tumor_Type', 'Survival_Rate', 'Tumor_Activity']
    X = df_clean.drop(columns=[col for col in features_to_drop if col in df_clean.columns])
    
    return X, tumor_type_target, survival_outcome, activity_target

# Apply preprocessing
X, y_tumor, y_survival, y_activity = preprocess_data(df)

print(f"\nFeatures shape: {X.shape}")
print(f"Tumor Type distribution:\n{y_tumor.value_counts()}")
print(f"Survival Outcome distribution (threshold=70):\n{y_survival.value_counts()}")
print(f"Tumor Activity distribution:\n{y_activity.value_counts()}")

# ============================
# TRAIN-TEST SPLIT
# ============================
X_train, X_test, y_train_tumor, y_test_tumor = train_test_split(
    X, y_tumor, test_size=0.2, stratify=y_tumor, random_state=42
)

# Also split for survival and activity predictions
_, _, y_train_survival, y_test_survival = train_test_split(
    X, y_survival, test_size=0.2, stratify=y_survival, random_state=42
)

_, _, y_train_activity, y_test_activity = train_test_split(
    X, y_activity, test_size=0.2, stratify=y_activity, random_state=42
)

# Scale features for SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining set size: {X_train.shape}")
print(f"Testing set size: {X_test.shape}")

# ============================
# TRAIN MULTIPLE MODELS
# ============================
def train_and_evaluate_model(model, model_name, X_train, X_test, y_train, y_test, scaled=False):
    """Train and evaluate a model"""
    if scaled:
        X_tr = X_train_scaled
        X_te = X_test_scaled
    else:
        X_tr = X_train
        X_te = X_test
    
    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)
    
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1-Score': f1_score(y_test, y_pred, average='weighted'),
    }
    
    return model, metrics, y_pred

# Initialize models
models = {
    'SVM': SVC(kernel='rbf', C=1.0, class_weight='balanced', probability=True, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

# Train for tumor type prediction
print("\n" + "="*60)
print("TUMOR TYPE PREDICTION MODELS")
print("="*60)

tumor_results = {}
for name, model in models.items():
    print(f"\n{'-'*40}")
    print(f"{name}")
    print(f"{'-'*40}")
    
    scaled = (name == 'SVM')
    trained_model, metrics, y_pred = train_and_evaluate_model(
        model, name, X_train, X_test, y_train_tumor, y_test_tumor, scaled
    )
    
    tumor_results[name] = {
        'model': trained_model,
        'metrics': metrics,
        'predictions': y_pred
    }
    
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

# ============================
# SURVIVAL PREDICTION MODEL
# ============================
print("\n" + "="*60)
print("SURVIVAL OUTCOME PREDICTION")
print("="*60)

# Train Random Forest for survival prediction
rf_survival = RandomForestClassifier(n_estimators=100, random_state=42)
rf_survival.fit(X_train, y_train_survival)
y_pred_survival = rf_survival.predict(X_test)

print(f"Survival Prediction Accuracy: {accuracy_score(y_test_survival, y_pred_survival):.4f}")

# ============================
# ACTIVITY PREDICTION MODEL
# ============================
print("\n" + "="*60)
print("TUMOR ACTIVITY PREDICTION")
print("="*60)

# Train Random Forest for activity prediction
rf_activity = RandomForestClassifier(n_estimators=100, random_state=42)
rf_activity.fit(X_train, y_train_activity)
y_pred_activity = rf_activity.predict(X_test)

print(f"Activity Prediction Accuracy: {accuracy_score(y_test_activity, y_pred_activity):.4f}")

# ============================
# MODEL COMPARISON
# ============================
print("\n" + "="*60)
print("MODEL COMPARISON - TUMOR TYPE PREDICTION")
print("="*60)

comparison_df = pd.DataFrame({
    'Model': list(tumor_results.keys()),
    'Accuracy': [tumor_results[m]['metrics']['Accuracy'] for m in tumor_results],
    'Precision': [tumor_results[m]['metrics']['Precision'] for m in tumor_results],
    'Recall': [tumor_results[m]['metrics']['Recall'] for m in tumor_results],
    'F1-Score': [tumor_results[m]['metrics']['F1-Score'] for m in tumor_results]
})

print(comparison_df.to_string(index=False))

# Select best model based on F1-Score
best_model_name = comparison_df.loc[comparison_df['F1-Score'].idxmax(), 'Model']
best_model = tumor_results[best_model_name]['model']
print(f"\nBest Model: {best_model_name}")

# ============================
# VISUALIZATIONS
# ============================
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Confusion Matrix for Best Model
ax1 = axes[0, 0]
cm = confusion_matrix(y_test_tumor, tumor_results[best_model_name]['predictions'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
ax1.set_title(f'{best_model_name} - Confusion Matrix')
ax1.set_xlabel('Predicted')
ax1.set_ylabel('Actual')

# 2. Feature Importance (Random Forest)
ax2 = axes[0, 1]
if 'Random Forest' in tumor_results:
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': tumor_results['Random Forest']['model'].feature_importances_
    })
    top_features = feature_importance.nlargest(10, 'importance')
    top_features.plot(kind='barh', x='feature', y='importance', ax=ax2, color='green')
    ax2.set_title('Top 10 Feature Importances (Random Forest)')

# 3. ROC Curve for SVM (if available)
ax3 = axes[0, 2]
if 'SVM' in tumor_results:
    svm_probs = tumor_results['SVM']['model'].predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test_tumor.map({'Malignant': 1, 'Benign': 0}), svm_probs)
    roc_auc = auc(fpr, tpr)
    ax3.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax3.plot([0, 1], [0, 1], 'k--')
    ax3.set_xlabel('False Positive Rate')
    ax3.set_ylabel('True Positive Rate')
    ax3.set_title('ROC Curve - SVM')
    ax3.legend()

# 4. Model Comparison Bar Chart
ax4 = axes[1, 0]
comparison_df.plot(kind='bar', x='Model', y=['Accuracy', 'F1-Score'], ax=ax4)
ax4.set_title('Model Performance Comparison')
ax4.set_ylabel('Score')
ax4.tick_params(axis='x', rotation=45)

# 5. Survival Rate Distribution
ax5 = axes[1, 1]
df['Survival_Rate'].hist(bins=30, ax=ax5, alpha=0.7)
ax5.axvline(x=70, color='red', linestyle='--', label='Threshold (70%)')
ax5.set_title('Survival Rate Distribution')
ax5.set_xlabel('Survival Rate (%)')
ax5.set_ylabel('Frequency')
ax5.legend()

# 6. Tumor Size vs Growth Rate
ax6 = axes[1, 2]
scatter = ax6.scatter(df['Tumor_Size'], df['Tumor_Growth_Rate'], 
                      c=df['Tumor_Type'].map({'Malignant': 'red', 'Benign': 'blue'}),
                      alpha=0.6)
ax6.set_title('Tumor Size vs Growth Rate')
ax6.set_xlabel('Tumor Size (cm)')
ax6.set_ylabel('Growth Rate (cm/month)')
ax6.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', 
                               markerfacecolor='red', markersize=10, label='Malignant'),
                    plt.Line2D([0], [0], marker='o', color='w', 
                               markerfacecolor='blue', markersize=10, label='Benign')])

plt.tight_layout()
plt.show()

# ============================
# SAVE MODELS AND PREPROCESSORS
# ============================
print("\n" + "="*60)
print("SAVING MODELS AND PREPROCESSORS")
print("="*60)

# Save the best tumor type model
joblib.dump(best_model, 'best_tumor_model.pkl')

# Save survival prediction model
joblib.dump(rf_survival, 'survival_model.pkl')

# Save activity prediction model
joblib.dump(rf_activity, 'activity_model.pkl')

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')

# Save feature names for reference
joblib.dump(list(X.columns), 'feature_names.pkl')

print("Models saved successfully!")
print(f"Best tumor model: {best_model_name}")
print(f"Models saved: best_tumor_model.pkl, survival_model.pkl, activity_model.pkl, scaler.pkl")