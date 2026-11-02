# Starbucks dataset: basic exploratory data analysis (EDA)
# Run from the project folder with: python ml.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("starbucks.csv")
df.columns = df.columns.str.strip()

print("--- Dataset shape ---")
print(df.shape)
print("\n--- Dataset information ---")
df.info()
print("\n--- First five rows ---")
print(df.head())
print("\n--- Missing values by column ---")
print(df.isnull().sum())
print("\n--- Numeric summary ---")
print(df.describe(include="all").transpose())

# Pie chart: top beverage categories
if "Beverage_category" in df.columns:
    counts = df["Beverage_category"].value_counts()
    plt.figure(figsize=(8, 8))
    plt.pie(
        counts,
        labels=counts.index,
        autopct="%1.1f%%",
        startangle=140,
        colors=sns.color_palette("pastel", len(counts)),
    )
    plt.title("Distribution of Starbucks Beverage Categories")
    plt.axis("equal")
    plt.tight_layout()
    plt.show()

# Bar chart: average calories by category
if {"Beverage_category", "Calories"}.issubset(df.columns):
    df["Calories"] = pd.to_numeric(df["Calories"], errors="coerce")
    averages = df.groupby("Beverage_category")["Calories"].mean().sort_values()
    plt.figure(figsize=(10, 5))
    averages.plot(kind="barh")
    plt.title("Average Calories by Beverage Category")
    plt.xlabel("Average calories")
    plt.ylabel("Beverage category")
    plt.tight_layout()
    plt.show()
