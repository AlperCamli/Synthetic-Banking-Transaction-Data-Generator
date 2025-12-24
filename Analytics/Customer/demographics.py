import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("customers.csv")

# Configure seaborn style
sns.set(style="whitegrid")

# Columns to exclude
exclude_cols = [
    "customer_id", "first_name", "last_name",
    "date_of_birth", "job_title", "bills_due_day", "rent_due_day"
]


# Loop through remaining columns
for col in df.columns:
    if col in exclude_cols:
        continue  # Skip excluded columns


    plt.figure(figsize=(8, 5))

    # Numeric columns → histogram
    if pd.api.types.is_numeric_dtype(df[col]):
        sns.histplot(df[col].dropna(), kde=True, bins=20, color="skyblue")
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Count")

    # Categorical / object columns → bar chart
    elif pd.api.types.is_object_dtype(df[col]) or df[col].dtype == "bool":
        order = df[col].value_counts().index
        sns.countplot(x=col, data=df, order=order, palette="Set2")
        plt.title(f"Distribution of {col}")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Count")

    plt.tight_layout()
    plt.show()
