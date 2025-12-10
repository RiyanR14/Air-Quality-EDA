import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

file_path = "city_day.csv"
df = pd.read_csv(r"C:\Users\HP\Downloads\Air-Quality-EDA\city_day.csv")


print("🔹 First 5 rows:")
print(df.head(), "\n")

print("🔹 Dataset Info:")
print(df.info(), "\n")

print("🔹 Summary Statistics:")
print(df.describe(include="all"), "\n")

print("🔹 Shape of Data (rows, columns):", df.shape)

if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

print("\n🔹 Missing Values per Column:")
print(df.isnull().sum())

missing_percent = df.isnull().mean() * 100
print("\n🔹 Missing Values Percentage:")
print(missing_percent)

numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

if "City" in df.columns:
    df = df.dropna(subset=["City"])

if "AQI" in df.columns:
    df = df.dropna(subset=["AQI"])

print("\n🔹 Shape after cleaning:", df.shape)

if "AQI" in df.columns:
    plt.figure(figsize=(8, 5))
    sns.histplot(df["AQI"], kde=True)
    plt.title("Distribution of AQI")
    plt.xlabel("AQI")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

if "City" in df.columns:
    plt.figure(figsize=(10, 6))
    df["City"].value_counts().head(10).plot(kind="bar")
    plt.title("Top 10 Cities by Number of Records")
    plt.xlabel("City")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

if "AQI_Bucket" in df.columns:
    plt.figure(figsize=(8, 5))
    df["AQI_Bucket"].value_counts().plot(kind="bar")
    plt.title("AQI Bucket Distribution")
    plt.xlabel("AQI Bucket")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

if {"AQI", "PM2.5"}.issubset(df.columns):
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x="PM2.5", y="AQI", alpha=0.5)
    plt.title("AQI vs PM2.5")
    plt.xlabel("PM2.5")
    plt.ylabel("AQI")
    plt.tight_layout()
    plt.show()

if {"AQI", "PM10"}.issubset(df.columns):
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x="PM10", y="AQI", alpha=0.5)
    plt.title("AQI vs PM10")
    plt.xlabel("PM10")
    plt.ylabel("AQI")
    plt.tight_layout()
    plt.show()

if {"City", "AQI"}.issubset(df.columns):
    city_aqi = df.groupby("City")["AQI"].mean().sort_values(ascending=False).head(10)
    plt.figure(figsize=(10, 6))
    city_aqi.plot(kind="bar")
    plt.title("Top 10 Cities by Average AQI")
    plt.xlabel("City")
    plt.ylabel("Average AQI")
    plt.tight_layout()
    plt.show()

if "Date" in df.columns and "AQI" in df.columns:
    df = df.sort_values("Date")
    daily_aqi = df.groupby("Date")["AQI"].mean()

    plt.figure(figsize=(12, 5))
    plt.plot(daily_aqi.index, daily_aqi.values)
    plt.title("Daily Average AQI Over Time (All Cities)")
    plt.xlabel("Date")
    plt.ylabel("Average AQI")
    plt.tight_layout()
    plt.show()

    example_city = df["City"].unique()[0]
    city_df = df[df["City"] == example_city]
    city_daily_aqi = city_df.groupby("Date")["AQI"].mean()

    plt.figure(figsize=(12, 5))
    plt.plot(city_daily_aqi.index, city_daily_aqi.values)
    plt.title(f"Daily Average AQI Over Time - {example_city}")
    plt.xlabel("Date")
    plt.ylabel("Average AQI")
    plt.tight_layout()
    plt.show()

if len(numeric_cols) > 1:
    plt.figure(figsize=(10, 8))
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=False, cmap="coolwarm")
    plt.title("Correlation Heatmap of Numeric Features")
    plt.tight_layout()
    plt.show()

if {"City", "AQI"}.issubset(df.columns):
    top_cities = df["City"].value_counts().head(5).index
    top_city_df = df[df["City"].isin(top_cities)]

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=top_city_df, x="City", y="AQI")
    plt.title("AQI Distribution for Top 5 Cities")
    plt.xlabel("City")
    plt.ylabel("AQI")
    plt.tight_layout()
    plt.show()

print("\n✅ EDA & Visualization completed.")
