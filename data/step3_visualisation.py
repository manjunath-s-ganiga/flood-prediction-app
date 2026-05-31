# ============================================================
# STEP 3: Exploratory Data Analysis & Visualisation
# ============================================================


import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np

df = pd.read_csv('data_cleaned_full.csv')

sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams.update({'figure.dpi': 120, 'axes.titlesize': 13,
                     'axes.labelsize': 11})

# ── 3.1  Severity Distribution ────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

counts = df['Severity'].value_counts()
axes[0].bar(counts.index, counts.values,
            color=['#4CAF50','#FF9800','#F44336'])
axes[0].set_title('Flood Severity Distribution')
axes[0].set_xlabel('Severity Level')
axes[0].set_ylabel('Count')
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 20, str(v), ha='center', fontweight='bold')

axes[1].pie(counts.values, labels=counts.index,
            colors=['#4CAF50','#FF9800','#F44336'],
            autopct='%1.1f%%', startangle=140)
axes[1].set_title('Severity Proportion')

plt.tight_layout()
plt.savefig('plot_01_severity_distribution.png', bbox_inches='tight')
plt.show()
print("✅ Plot 1: Severity distribution saved.")

# ── 3.2  Rainfall by Severity ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

sns.boxplot(data=df, x='Severity', y='Rainfall_Quantity_mm',
            palette=['#4CAF50','#FF9800','#F44336'],
            order=['Low','Medium','High'], ax=axes[0])
axes[0].set_title('Rainfall vs Severity')

sns.violinplot(data=df, x='Severity', y='Water_Level_m',
               palette=['#4CAF50','#FF9800','#F44336'],
               order=['Low','Medium','High'], ax=axes[1])
axes[1].set_title('Water Level vs Severity')

plt.tight_layout()
plt.savefig('plot_02_rainfall_water_severity.png', bbox_inches='tight')
plt.show()
print("✅ Plot 2: Rainfall & water level plots saved.")

# ── 3.3  Monthly Flood Frequency ──────────────────────────────
month_order = ['January','February','March','April','May','June',
               'July','August','September','October','November','December']
monthly = df['Month'].value_counts().reindex(month_order).fillna(0)

fig, ax = plt.subplots(figsize=(13, 4))
bars = ax.bar(monthly.index, monthly.values,
              color=sns.color_palette('Blues_d', 12))
ax.set_title('Flood Events by Month')
ax.set_xlabel('Month')
ax.set_ylabel('Number of Events')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig('plot_03_monthly_flood_frequency.png', bbox_inches='tight')
plt.show()
print("✅ Plot 3: Monthly frequency saved.")

# ── 3.4  Correlation Heatmap ──────────────────────────────────
numeric_cols = ['Rainfall_Quantity_mm','Flood_Duration_Days',
                'Water_Level_m','Affected_People',
                'Affected_Area_sqkm','Severity_Enc']
corr = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, ax=ax, linewidths=0.5)
ax.set_title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('plot_04_correlation_heatmap.png', bbox_inches='tight')
plt.show()
print("✅ Plot 4: Heatmap saved.")

# ── 3.5  Top 10 Most Affected Locations ───────────────────────
top10 = (df.groupby('Location')['Affected_People']
           .mean().nlargest(10).reset_index())

fig, ax = plt.subplots(figsize=(10, 4))
sns.barplot(data=top10, y='Location', x='Affected_People',
            palette='Reds_r', ax=ax)
ax.set_title('Top 10 Locations – Avg Affected People')
ax.set_xlabel('Average Affected People')
plt.tight_layout()
plt.savefig('plot_05_top10_locations.png', bbox_inches='tight')
plt.show()
print("✅ Plot 5: Top locations saved.")

# ── 3.6  Pairplot of key features ────────────────────────────
pair_cols = ['Rainfall_Quantity_mm','Water_Level_m',
             'Flood_Duration_Days','Severity']
g = sns.pairplot(df[pair_cols], hue='Severity',
                 palette={'Low':'#4CAF50','Medium':'#FF9800','High':'#F44336'},
                 plot_kws={'alpha': 0.4, 's': 15})
g.fig.suptitle('Pairplot – Key Features', y=1.02)
plt.savefig('plot_06_pairplot.png', bbox_inches='tight')
plt.show()
print("✅ Plot 6: Pairplot saved.")

print("\n🎉 All visualisations complete!")
