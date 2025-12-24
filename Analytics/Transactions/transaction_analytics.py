# === Transaction Analytics & Customer Visuals ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# Config
# -----------------------------
TRANSACTIONS_CSV = "transactions.csv"
MERCHANTS_CSV    = "merchants.csv"     # optional (for category enrichment)
CHANNELS_CSV     = "channels.csv"      # optional (for channel names)
CATEGORIES_CSV   = "categories.csv"    # optional (for category names)

# Your mapping from generator (adjust if different in your data)
# 1 = debit, 2 = credit
TX_TYPE_SIGN = {1: -1, 2: 1}

# -----------------------------
# Load
# -----------------------------
df = pd.read_csv(TRANSACTIONS_CSV, parse_dates=["transaction_date"])

# Basic safety: ensure required cols exist
required = {
    "transaction_id_dd","transaction_date","transaction_date_key","customer_id",
    "account_id","merchant_id","branch_id","channel_id","transaction_type_id",
    "amount","currency","is_recurring","previous_balance","balance_after_transaction"
}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"transactions.csv missing columns: {missing}")

# -----------------------------
# Enrich (optional: names for channels and categories)
# -----------------------------
channels = None
if Path(CHANNELS_CSV).exists():
    channels = pd.read_csv(CHANNELS_CSV)
    if {"channel_id","channel_name"}.issubset(channels.columns):
        df = df.merge(channels[["channel_id","channel_name"]], on="channel_id", how="left")

# To get category_name we can join: transactions -> merchants -> categories
if Path(MERCHANTS_CSV).exists() and Path(CATEGORIES_CSV).exists():
    merchants  = pd.read_csv(MERCHANTS_CSV)
    categories = pd.read_csv(CATEGORIES_CSV)
    if {"merchant_id","category_id"}.issubset(merchants.columns) and \
       {"category_id","category_name"}.issubset(categories.columns):
        df = df.merge(merchants[["merchant_id","category_id"]], on="merchant_id", how="left")
        df = df.merge(categories[["category_id","category_name"]], on="category_id", how="left")

# -----------------------------
# Derivations
# -----------------------------
# Signed amount: credits positive, debits negative
df["signed_amount"] = df["transaction_type_id"].map(TX_TYPE_SIGN).fillna(-1) * df["amount"].astype(float)

# For daily plots, collapse to date (no time)
df["date"] = df["transaction_date"].dt.date

# -----------------------------
# Overall visuals
# -----------------------------

# 1) Daily transaction count
daily_cnt = df.groupby("date").size()
plt.figure()
daily_cnt.plot()
plt.title("Daily Transaction Count")
plt.xlabel("Date")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# 2) Daily debit total
daily_debit = df.loc[df["transaction_type_id"]==1].groupby("date")["amount"].sum()
plt.figure()
daily_debit.plot()
plt.title("Daily Debit Amount (Sum)")
plt.xlabel("Date")
plt.ylabel("USD")
plt.tight_layout()
plt.show()

# 3) Daily credit total
daily_credit = df.loc[df["transaction_type_id"]==2].groupby("date")["amount"].sum()
plt.figure()
daily_credit.plot()
plt.title("Daily Credit Amount (Sum)")
plt.xlabel("Date")
plt.ylabel("USD")
plt.tight_layout()
plt.show()

# 4) Transaction amount distribution (absolute amounts)
plt.figure()
df["amount"].astype(float).plot(kind="hist", bins=50)
plt.title("Transaction Amount Distribution")
plt.xlabel("USD")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# 5) Channel distribution (if channel_name available; else by channel_id)
if "channel_name" in df.columns:
    ch_series = df["channel_name"].value_counts().sort_values(ascending=False)
else:
    ch_series = df["channel_id"].value_counts().sort_values(ascending=False)

plt.figure()
ch_series.plot(kind="bar")
plt.title("Channel Distribution (by Count)")
plt.xlabel("Channel")
plt.ylabel("Transactions")
plt.tight_layout()
plt.show()

# 6) Category distribution (if category_name available)
if "category_name" in df.columns:
    cat_series = df["category_name"].value_counts().head(20)  # top 20
    plt.figure()
    cat_series.plot(kind="bar")
    plt.title("Top Categories (by Count)")
    plt.xlabel("Category")
    plt.ylabel("Transactions")
    plt.tight_layout()
    plt.show()

