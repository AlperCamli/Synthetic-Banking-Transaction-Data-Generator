import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date

# Load dataset
df = pd.read_csv("customers.csv")

# Convert date_of_birth to datetime
df['date_of_birth'] = pd.to_datetime(df['date_of_birth'], errors='coerce')

# Calculate age
today = date.today()
df['age'] = df['date_of_birth'].apply(lambda dob: today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day)) if pd.notnull(dob) else None)

# Scatter plot: Salary vs Age
plt.figure(figsize=(8,6))
sns.scatterplot(x="age", y="monthly_salary_usd", hue="segment", data=df, alpha=0.7)
plt.title("Monthly Salary vs Age (by Segment)")
plt.xlabel("Age")
plt.ylabel("Monthly Salary (USD)")
plt.legend(title="Segment")
plt.show()
