# -----------------------------
# Customer-specific visuals
# -----------------------------
def plot_customer_views(transactions: pd.DataFrame, customer_id):
    """
    Plots:
      - Balance after each transaction over time (by event sequence)
      - Signed transaction amounts (debit negative, credit positive) by sequence
    """
    cdf = transactions.loc[transactions["customer_id"] == customer_id].copy()
    if cdf.empty:
        print(f"No transactions found for {customer_id}")
        return

    # Sort by date then by transaction_id for stable sequence
    cdf = cdf.sort_values(["transaction_date", "transaction_id_dd"]).reset_index(drop=True)
    cdf["event_idx"] = np.arange(len(cdf))

    # Line: balance after transaction by event sequence
    plt.figure()
    plt.plot(cdf["event_idx"], cdf["balance_after_transaction"].astype(float))
    plt.title(f"Balance Over Transactions — {customer_id}")
    plt.xlabel("Transaction # (ordered by time)")
    plt.ylabel("Balance After Transaction (USD)")
    plt.tight_layout()
    plt.show()

    """# Bar: signed amount by event sequence
    plt.figure()
    plt.bar(cdf["event_idx"], cdf["signed_amount"].astype(float))
    plt.title(f"Transaction Amounts (Signed) — {customer_id}")
    plt.xlabel("Transaction # (ordered by time)")
    plt.ylabel("Signed Amount (USD)")
    plt.tight_layout()
    plt.show()"""

    """# Optional: daily balance (last balance per day)
    daily_last = (cdf.groupby(cdf["transaction_date"].dt.date)["balance_after_transaction"]
                    .last())
    plt.figure()
    daily_last.plot()
    plt.title(f"Daily Ending Balance — {customer_id}")
    plt.xlabel("Date")
    plt.ylabel("Balance (USD)")
    plt.tight_layout()
    plt.show()"""

# Example usage for a specific customer:
plot_customer_views(df, "CUST0090")   # <-- change to the ID you want
