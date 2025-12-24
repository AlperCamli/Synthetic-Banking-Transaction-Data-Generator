import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
df = pd.read_csv("customers.csv")

# --- Basic Exploration ---
print(df.head())          # Show first rows
print(df.info())          # Data types
print(df.describe())      # Summary statistics
print(df.isna().sum())    # Missing values check

# --- Distribution of Monthly Income ---
plt.figure(figsize=(8,5))
sns.histplot(df['monthly_salary_usd'], bins=40, kde=True, color="skyblue")
plt.title("Distribution of Monthly Income")
plt.xlabel("Monthly Income")
plt.ylabel("Count")
plt.show()

# --- Distribution of Credit Scores ---
plt.figure(figsize=(8,5))
sns.histplot(df['credit_score'], bins=20, kde=True, color="skyblue")
plt.title("Distribution of Credit Scores")
plt.xlabel("Credit Score")
plt.ylabel("Count")
plt.show()

# --- Income vs. Credit Score ---
plt.figure(figsize=(8,5))
sns.scatterplot(x="annual_income", y="credit_score", hue="segment", data=df, alpha=0.7)
plt.title("Annual Income vs. Credit Score by Segment")
plt.show()

# --- Average Monthly Salary by Segment ---
plt.figure(figsize=(8,5))
sns.barplot(x="segment", y="monthly_salary_usd", data=df, estimator="mean", ci=None)
plt.xticks(rotation=45)
plt.title("Average Monthly Salary by Segment")
plt.show()

# --- Housing Distribution ---
plt.figure(figsize=(6,4))
sns.countplot(x="housing", data=df, palette="Set2")
plt.title("Housing Status Distribution")
plt.show()

# --- Weekly Transaction Rate by Segment ---
plt.figure(figsize=(8,5))
sns.boxplot(x="segment", y="weekly_txn_rate", data=df)
plt.xticks(rotation=45)
plt.title("Weekly Transaction Rate by Segment")
plt.show()

# --- Spending Channels Breakdown ---
channel_cols = ["w_channel_POS", "w_channel_online", "w_channel_ATM", "w_channel_branch"]
df[channel_cols].mean().plot(kind="bar", figsize=(6,4), color="coral")
plt.title("Average Transaction Channel Shares")
plt.ylabel("Average Share")
plt.show()

# --- Spending Categories Breakdown ---
cat_cols = [col for col in df.columns if col.startswith("w_cat_")]
df[cat_cols].mean().plot(kind="bar", figsize=(10,5), color="teal")
plt.title("Average Spending by Category")
plt.ylabel("Average Share")
plt.xticks(rotation=45)
plt.show()
