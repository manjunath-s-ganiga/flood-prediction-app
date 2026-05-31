# ============================================================
# STEP 1: Import Libraries & Load Data
# ============================================================

import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np

# ── Load Dataset ──────────────────────────────────────────────
df = pd.read_csv('india_flood_prediction_dataset_10000.csv')

print("=" * 60)
print("  FLOOD PREDICTION DATASET - INITIAL EXPLORATION")
print("=" * 60)

# Shape
print(f"\n📦 Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")

# Column names & dtypes
print("\n📋 Columns & Data Types:")
print(df.dtypes.to_string())

# First 5 rows
print("\n🔍 First 5 Rows:")
print(df.head().to_string())

# Basic statistics
print("\n📊 Statistical Summary (Numeric Columns):")
print(df.describe().to_string())

# Null values
print("\n❓ Null Value Count per Column:")
print(df.isnull().sum().to_string())

# Target distribution
print("\n🎯 Target Variable (Severity) Distribution:")
print(df['Severity'].value_counts())
print(f"\nPercentage:\n{df['Severity'].value_counts(normalize=True).mul(100).round(2).to_string()} %")

# Unique locations
print(f"\n  Unique Locations: {df['Location'].nunique()}")
print(f"Sample Locations: {df['Location'].unique()[:10].tolist()}")
