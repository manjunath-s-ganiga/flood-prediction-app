# ============================================================
# STEP 4: Train-Test Split & Random Forest Model Training
# ============================================================


import joblib
import warnings
warnings.filterwarnings('ignore')
import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np

from sklearn.model_selection  import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble         import RandomForestClassifier
from sklearn.preprocessing    import LabelEncoder
from sklearn.metrics          import (accuracy_score, classification_report,
                                      confusion_matrix, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt
import seaborn as sns

# ── 4.1  Load model-ready data ────────────────────────────────
df = pd.read_csv('data_model_ready.csv')

print("=" * 60)
print("  STEP 4: RANDOM FOREST – TRAINING & EVALUATION")
print("=" * 60)
print(f"\n📦 Dataset shape: {df.shape}")
print(f"   Columns: {df.columns.tolist()}")

# ── 4.2  Features & Target ────────────────────────────────────
FEATURE_COLS = [
    'Rainfall_Quantity_mm', 'Month_Num',
    'Flood_Duration_Days', 'Water_Level_m',
    'Affected_People', 'Affected_Area_sqkm',
    'Location_Enc',
    'Start_Month', 'Start_DayOfWeek',
    'End_Month', 'End_DayOfYear'
]

X = df[FEATURE_COLS]
y = df['Severity_Enc']   # 0=Low, 1=Medium, 2=High

print(f"\n✅ Features : {FEATURE_COLS}")
print(f"✅ Target   : Severity_Enc  (0=Low, 1=Medium, 2=High)")

# ── 4.3  Train / Test Split ───────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n🔀 Train size : {X_train.shape[0]}")
print(f"   Test size  : {X_test.shape[0]}")

# ── 4.4  Train Random Forest ──────────────────────────────────
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

print("\n⏳ Training Random Forest …")
rf.fit(X_train, y_train)
print("✅ Training complete!")

# ── 4.5  Cross-Validation (5-fold) ───────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf, X_train, y_train,
                             cv=cv, scoring='accuracy')
print(f"\n📈 5-Fold CV Accuracy : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"   Fold Scores       : {[round(s,4) for s in cv_scores]}")

# ── 4.6  Test Set Evaluation ──────────────────────────────────
y_pred = rf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"\n🎯 Test Accuracy : {acc:.4f} ({acc*100:.2f}%)")

label_names = ['Low', 'Medium', 'High']
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_names))

# ── 4.7  Confusion Matrix Plot ────────────────────────────────
cm  = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=label_names)
disp.plot(cmap='Blues', ax=ax, colorbar=False)
ax.set_title(f'Confusion Matrix  (Accuracy: {acc*100:.1f}%)')
plt.tight_layout()
plt.savefig('plot_07_confusion_matrix.png', bbox_inches='tight')
plt.show()
print("✅ Confusion matrix saved.")

# ── 4.8  Feature Importance Plot ─────────────────────────────
importances = pd.Series(rf.feature_importances_,
                         index=FEATURE_COLS).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(8, 5))
importances.plot(kind='barh', ax=ax, color='steelblue')
ax.set_title('Random Forest – Feature Importances')
ax.set_xlabel('Importance Score')
plt.tight_layout()
plt.savefig('plot_08_feature_importance.png', bbox_inches='tight')
plt.show()
print("✅ Feature importance plot saved.")

# ── 4.9  Save Model & Feature List ───────────────────────────
joblib.dump(rf, 'random_forest_model.pkl')
joblib.dump(FEATURE_COLS, 'feature_cols.pkl')

print("\n💾 Model saved → random_forest_model.pkl")
print("💾 Features   → feature_cols.pkl")
print("\n🎉 Training pipeline complete!")
