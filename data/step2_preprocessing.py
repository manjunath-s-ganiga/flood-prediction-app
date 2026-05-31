# ============================================================
# STEP 2: Data Preprocessing & Cleaning
# ============================================================

import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np

df = pd.read_csv('india_flood_prediction_dataset_10000.csv')

print("=" * 60)
print("  STEP 2: DATA PREPROCESSING & CLEANING")
print("=" * 60)

# ── 2.1  Keep Start_Date & End_Date as NaN (as required) ─────
# The user wants these columns to retain their null values as-is.
# We'll just parse them for feature extraction, not impute them.

# Convert date columns — invalid/missing → NaT (which shows as NaN)
df['Start_Date'] = pd.to_datetime(df['Start_Date'], errors='coerce')
df['End_Date']   = pd.to_datetime(df['End_Date'],   errors='coerce')

print("\n✅ Start_Date & End_Date converted to datetime.")
print(f"   Null Start_Date : {df['Start_Date'].isnull().sum()}")
print(f"   Null End_Date   : {df['End_Date'].isnull().sum()}")

# ── 2.2  Extract Date Features (only where not null) ──────────
df['Start_Month']    = df['Start_Date'].dt.month       # NaN where date is null
df['Start_DayOfWeek']= df['Start_Date'].dt.dayofweek   # NaN where date is null
df['End_Month']      = df['End_Date'].dt.month
df['End_DayOfYear']  = df['End_Date'].dt.dayofyear

print("\n✅ Date-derived features extracted (NaN preserved for null rows).")

# ── 2.3  Encode 'Month' (string) → ordinal ────────────────────
month_order = {
    'January':1,'February':2,'March':3,'April':4,
    'May':5,'June':6,'July':7,'August':8,
    'September':9,'October':10,'November':11,'December':12
}
df['Month_Num'] = df['Month'].map(month_order)

# ── 2.4  Encode categorical columns ───────────────────────────
# Location: label-encode (saves a mapping dict for inference)
from sklearn.preprocessing import LabelEncoder

le_location = LabelEncoder()
df['Location_Enc'] = le_location.fit_transform(df['Location'])

# Severity: encode target
severity_map = {'Low': 0, 'Medium': 1, 'High': 2}
df['Severity_Enc'] = df['Severity'].map(severity_map)

print("\n✅ Categorical encoding done.")
print(f"   Severity mapping : {severity_map}")

# ── 2.5  Drop raw string/date columns for modelling ───────────
# We keep the originals in `df` for display but create a clean copy for ML
drop_cols = ['Location', 'Month', 'Start_Date', 'End_Date', 'Severity']
df_model  = df.drop(columns=drop_cols)

print(f"\n✅ Model DataFrame columns: {df_model.columns.tolist()}")

# ── 2.6  Final null check on model data ───────────────────────
print("\n❓ Null counts in model DataFrame:")
print(df_model.isnull().sum().to_string())

# We fill remaining NaN in extracted date features with -1 (sentinel)
# so tree-based models handle them gracefully
date_feat_cols = ['Start_Month','Start_DayOfWeek','End_Month','End_DayOfYear']
df_model[date_feat_cols] = df_model[date_feat_cols].fillna(-1)

print("\n✅ Date feature nulls filled with -1 (sentinel for tree models).")

# ── 2.7  Save cleaned data ─────────────────────────────────────
df.to_csv('data_cleaned_full.csv', index=False)
df_model.to_csv('data_model_ready.csv', index=False)

print("\n💾 Saved: data_cleaned_full.csv  &  data_model_ready.csv")
print(f"\n📦 Model-ready shape: {df_model.shape}")
print(df_model.head())
