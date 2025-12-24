import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
customers = pd.read_csv("customers.csv")

# Compute monthly income = salary + passive + rent_in
customers["monthly_income_usd"] = (
    customers["monthly_salary_usd"] +
    customers["monthly_passive_income_usd"] +
    customers["monthly_rent_in_usd"]
)

# Define bins (income ranges)
bins = np.linspace(customers["monthly_income_usd"].min(),
                   customers["monthly_income_usd"].max(), 20)

# Bin customers and sum their income per bin
bin_labels = pd.cut(customers["monthly_income_usd"], bins=bins)
total_income_distribution = customers.groupby(bin_labels)["monthly_income_usd"].sum()

# Plot Total Income Distribution
plt.figure(figsize=(12,6))
total_income_distribution.plot(kind="bar", color="steelblue", edgecolor="black")
plt.title("Total Monthly Income Distribution")
plt.xlabel("Monthly Income Range (USD)")
plt.ylabel("Total Income of Customers in Range (USD)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
