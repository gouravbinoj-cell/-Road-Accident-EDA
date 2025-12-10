
import pandas as pd
import numpy as np
print("=== Milestone 1: Dataset Loading & Basic Exploration ===")
df = pd.read_csv("US_Accidents_March23.csv")
print("\n--- Head of Dataset ---")
print(df.head())
print("\n--- Shape of Dataset ---")
print(df.shape)
print("\n--- Columns ---")
print(df.columns)
print("\n--- Info ---")
print(df.info())
print("\n--- Describe ---")
print(df.describe())
print("\n--- Missing Values ---")
print(df.isnull().sum().sort_values(ascending=False))


print("=== Milestone 1 – Week 2: Data Cleaning & Preprocessing ===")

df = pd.read_csv("US_Accidents_March23.csv")

missing_percent = (df.isnull().sum() / len(df)) * 100

cols_to_drop = missing_percent[missing_percent > 40].index
df.drop(columns=cols_to_drop, inplace=True)

print("\nDropped columns (too many missing values):")
print(cols_to_drop)

df["Start_Time"] = pd.to_datetime(df["Start_Time"], errors="coerce")

df = df.dropna(subset=["Start_Time"])

print("\nConverted Start_Time column to datetime.")


df["Hour"] = df["Start_Time"].dt.hour
df["Weekday"] = df["Start_Time"].dt.day_name()
df["Month"] = df["Start_Time"].dt.month_name()

print("\nCreated new features: Hour, Weekday, Month")


before = len(df)
df.drop_duplicates(inplace=True)
after = len(df)

print(f"\nRemoved {before - after} duplicate rows")


df = df[df["Distance(mi)"] < df["Distance(mi)"].quantile(0.99)]
print("\nHandled outliers for Distance (removed top 1%).")


df.to_csv("cleaned_week2.csv", index=False)
print("\nSaved cleaned dataset as cleaned_week2.csv")
