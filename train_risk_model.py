import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import shap

# 1. LOAD DATA
df = pd.read_csv('credit_risk_dataset.csv')

# 2. PREPROCESSING
# Encode City_Tier (Tier 1=0, Tier 2=1...)
le = LabelEncoder()
df['City_Tier_Encoded'] = le.fit_transform(df['City_Tier'])

# Drop ID and original categorical column
X = df.drop(['Applicant_ID', 'Default_Status', 'City_Tier'], axis=1)
y = df['Default_Status']

# Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Original Training Defaults: {sum(y_train)} / {len(y_train)}")

# 3. APPLY SMOTE (The "Magic" Step)
# Synthetically generate defaulters to balance the dataset
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"Resampled Training Defaults: {sum(y_train_resampled)} / {len(y_train_resampled)} (Balanced!)")

# 4. TRAIN MODEL (XGBoost)
# We use scale_pos_weight for extra safety, though SMOTE helps a lot
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_train_resampled, y_train_resampled)

# 5. EVALUATE
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n--- MODEL PERFORMANCE REPORT ---")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

# 6. VISUALIZATION: CONFUSION MATRIX
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
            xticklabels=['Paid', 'Default'], yticklabels=['Paid', 'Default'])
plt.title('Confusion Matrix (SMOTE Enhanced)')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.show()

# 7. EXPLAINABILITY (SHAP)
# Explain why the model makes decisions (Crucial for regulations)
explainer = shap.Explainer(model)
shap_values = explainer(X_test)

plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
plt.title("Key Risk Drivers (SHAP Values)")
plt.tight_layout()
plt.show()