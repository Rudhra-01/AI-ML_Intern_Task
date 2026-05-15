import csv
import json
import numpy as np
import pandas as pd
# Load CSV using open() and print first 5 rows
print("\n Load CSV using open() and print first 5 rows \n")
with open("Titanic-Dataset.csv", "r", encoding="utf-8") as file:
    reader = csv.reader(file)
    rows = list(reader)
    print("First 5 Rows:\n")
    for row in rows[:6]:   # Including header
        print(row)
# Filter survived passengers and write to survivors.csv
print("\nFilter survived passengers and write to survivors.csv\n")
with open("Titanic-Dataset.csv", "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    survived_rows = []
    for row in reader:
        if row["Survived"] == "1":
            survived_rows.append(row)
# Write survivors to new CSV
with open("survivors.csv", "w", newline="", encoding="utf-8") as file:
    fieldnames = survived_rows[0].keys()
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(survived_rows)
print("survivors.csv file created successfully!")
# Convert CSV to JSON--
print("\nConvert CSV to JSON\n")
data = []
with open("Titanic-Dataset.csv", "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    for row in reader:
        data.append(row)
with open("titanic.json", "w", encoding="utf-8") as json_file:
    json.dump(data, json_file, indent=4)
print("titanic.json file created successfully!")
# PART 2 — NUMPY
# Load dataset using pandas for NumPy tasks
df = pd.read_csv("Titanic-Dataset.csv")
# Age column analysis
print("\nLoad dataset using pandas for NumPy tasks\n")
age_array = df["Age"].to_numpy()
# Calculate mean without NaN
mean_age = np.nanmean(age_array)
# Replace NaN with mean
age_array = np.where(np.isnan(age_array), mean_age, age_array)
# Statistics
mean_value = np.mean(age_array)
median_value = np.median(age_array)
std_value = np.std(age_array)
print("Mean Age:", mean_value)
print("Median Age:", median_value)
print("Standard Deviation:", std_value)
# Age group classification
print("\nAge group classification\n")
age_groups = np.where(
    age_array < 18,
    "Child",
    np.where(age_array <= 60, "Adult", "Senior")
)
# Count groups
unique, counts = np.unique(age_groups, return_counts=True)
print("Passenger Age Groups:\n")
for group, count in zip(unique, counts):
    print(group, ":", count)
# Fare normalization
print("\nFare normalization\n")
fare_array = df["Fare"].to_numpy()
# Min-Max Normalization
normalized_fare = (
    (fare_array - np.min(fare_array)) /
    (np.max(fare_array) - np.min(fare_array))
)
print("First 10 Normalized Fare Values:\n")
print(normalized_fare[:10])
# PART 3 — PANDAS
# DataFrame information
print("\nDataFrame information\n")
print("Shape of Dataset:")
print(df.shape)
print("\nColumn Names:")
print(df.columns)
print("\nData Types:")
print(df.dtypes)
print("\nMissing Values:")
print(df.isnull().sum())
# Group by Pclass
print("\nGroup by Pclass\n")
grouped = df.groupby("Pclass").agg({
    "Survived": "mean",
    "Age": "mean",
    "Fare": "mean"
})
print(grouped)
# Family Size and Survival Rate
print("\nFamily Size and Survival Rate\n")
df["FamilySize"] = df["SibSp"] + df["Parch"]
family_survival = (
    df.groupby("FamilySize")["Survived"]
    .mean()
    .sort_values(ascending=False)
)
print(family_survival)
# Filter first class women
print("\nFilter first class women\n")
filtered_df = df[
    (df["Sex"] == "female") &
    (df["Age"] >= 18) &
    (df["Age"] <= 35) &
    (df["Pclass"] == 1)
]
filtered_df.to_csv("first_class_women.csv", index=False)
print("first_class_women.csv created successfully!")
print("\nAll Tasks Completed Successfully!")