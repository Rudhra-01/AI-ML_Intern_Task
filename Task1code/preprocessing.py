import pandas as pd
import numpy as np
df = pd.read_csv("Titanic-Dataset.csv")

print("\n========== ORIGINAL DATASET ==========\n")
print(df.head())
# 1. CHECK MISSING VALUES
print("\n========== MISSING VALUES ==========\n")
print(df.isnull().sum())
# 2. HANDLE MISSING VALUES
print("\n========== HANDLING MISSING VALUES ==========\n")
# Fill missing Age values with mean
mean_age = df["Age"].mean()
df["Age"] = df["Age"].fillna(mean_age)
print("Missing Age values filled with mean age.")
# Fill missing Embarked values with mode
mode_embarked = df["Embarked"].mode()[0]
df["Embarked"] = df["Embarked"].fillna(mode_embarked)
print("Missing Embarked values filled with mode.")
# Drop Cabin column because many values are missing
df.drop(columns=["Cabin"], inplace=True)
print("Cabin column dropped.")
# 3. CHECK DATASET AFTER CLEANING
print("\n========== DATASET AFTER CLEANING ==========\n")
print(df.isnull().sum())
# 4. FEATURE ENGINEERING
print("\n========== FEATURE ENGINEERING ==========\n")
# Create FamilySize feature
df["FamilySize"] = df["SibSp"] + df["Parch"]
print("FamilySize feature created.")
# Create IsAlone feature
df["IsAlone"] = np.where(df["FamilySize"] == 0, 1, 0)
print("IsAlone feature created.")
# 5. AGE GROUP CLASSIFICATION
print("\n========== AGE GROUP CLASSIFICATION ==========\n")
df["AgeGroup"] = np.where(
    df["Age"] < 18,
    "Child",
    np.where(df["Age"] <= 60, "Adult", "Senior")
)
print(df[["Age", "AgeGroup"]].head())
# 6. NORMALIZE FARE COLUMN
print("\n========== FARE NORMALIZATION ==========\n")
df["Fare_Normalized"] = (
    (df["Fare"] - df["Fare"].min())
    /
    (df["Fare"].max() - df["Fare"].min())
)
print(df[["Fare", "Fare_Normalized"]].head())
# 7. ENCODE CATEGORICAL DATA
print("\n========== ENCODING CATEGORICAL DATA ==========\n")
# Convert Sex column into numbers
# Male = 0
# Female = 1
df["Sex"] = df["Sex"].map({
    "male": 0,
    "female": 1
})
print("Sex column encoded.")
# Encode Embarked column
df["Embarked"] = df["Embarked"].map({
    "S": 0,
    "C": 1,
    "Q": 2
})
print("Embarked column encoded.")
# 8. FINAL DATASET INFORMATION
print("\n========== FINAL DATASET INFO ==========\n")
print(df.head())
print("\nDataset Shape:")
print(df.shape)
print("\nData Types:")
print(df.dtypes)
# 9. EXPORT PREPROCESSED DATASET
df.to_csv("preprocessed_titanic.csv", index=False)
print("\npreprocessed_titanic.csv created successfully!")
# FINAL MESSAGE
print("\n========== DATA PREPROCESSING COMPLETED ==========")