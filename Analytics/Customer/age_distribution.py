import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime

# Assuming df is your customers DataFrame
df = pd.read_csv("customers.csv")

# Compute age
today = datetime.today()
df["age"] = pd.to_datetime(df["date_of_birth"]).apply(lambda dob: (today - dob).days // 365)

# Plot histogram of ages
plt.figure(figsize=(10,6))
sns.histplot(df["age"], bins=30, kde=True, color="skyblue")
plt.title("Age Distribution of Customers")
plt.xlabel("Age")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()
