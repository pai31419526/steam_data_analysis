import pandas as pd
import numpy as np
from scipy import stats

# 1. Load Data
file_path = r"C:\Users\Xinyi\OneDrive\Desktop\main_df2.csv"
df = pd.read_csv(file_path)

print("--- ADVANCED BUSINESS HYPOTHESIS TESTING ---\n")

# =============================================================================
# HYPOTHESIS 1: Price as a Quality Signal
# =============================================================================
# Comparing "Budget" (<$5) vs "Premium Indie" ($15-$30)
budget = df[df['price_usd'] < 5]['estimated_sales']
premium_indie = df[(df['price_usd'] >= 15) & (df['price_usd'] <= 30)]['estimated_sales']

# Using Mann-Whitney U to see if Premium sells MORE than Budget
u1, p1 = stats.mannwhitneyu(premium_indie, budget, alternative='greater')

print(f"[Test 1] Price Psychology")
print(f" - Budget (<$5) Median Sales: {budget.median():,.0f}")
print(f" - Premium ($15-$30) Median Sales: {premium_indie.median():,.0f}")
print(f" - p-value: {p1:.4e}")
if p1 < 0.05:
    print(" >>> INSIGHT: Higher prices correlate with higher sales volume. Price signals quality.")
else:
    print(" >>> INSIGHT: Cheap games still rule the volume market.")
print("-" * 50)

# =============================================================================
# HYPOTHESIS 2: The Indie Sanctuary (Level Playing Field)
# =============================================================================
# Filter only for Indie games
indie_df = df[df['genre_indie'] == 1]
small_indie = indie_df[indie_df['dev_game_count'] <= 2]['estimated_sales']
large_indie = indie_df[indie_df['dev_game_count'] > 10]['estimated_sales']

# Two-sided test: Is there ANY significant difference?
u2, p2 = stats.mannwhitneyu(small_indie, large_indie)

print(f"[Test 2] Indie Sanctuary Effect")
print(f" - Small Indie Median Sales: {small_indie.median():,.0f}")
print(f" - Veteran Indie Median Sales: {large_indie.median():,.0f}")
print(f" - p-value: {p2:.4f}")
if p2 > 0.05:
    print(" >>> INSIGHT: Success! In the Indie category, fame doesn't guarantee more sales.")
else:
    print(" >>> INSIGHT: Even in Indie, the 'Brand Halo' persists.")
print("-" * 50)

# =============================================================================
# HYPOTHESIS 3: Localization Multiplier
# =============================================================================
# Threshold: Is > 8 languages significantly better than 1-3 languages?
minimal_lang = df[df['supported_languages'] <= 3]['estimated_sales']
global_lang = df[df['supported_languages'] > 8]['estimated_sales']

u3, p3 = stats.mannwhitneyu(global_lang, minimal_lang, alternative='greater')

print(f"[Test 3] Localization Leverage")
print(f" - Minimal Lang (<=3) Median Sales: {minimal_lang.median():,.0f}")
print(f" - Global Lang (>8) Median Sales: {global_lang.median():,.0f}")
print(f" - p-value: {p3:.4e}")
if p3 < 0.05:
    print(" >>> INSIGHT: Localization is a massive multiplier. Go global or go home.")
else:
    print(" >>> INSIGHT: Language support has diminishing returns.")