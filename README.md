# Synthetic Banking Transaction Data Generator

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Project Overview
This project is a sophisticated synthetic data generator designed to simulate realistic banking transaction environments. It was developed as part of an internship project at **VakıfBank Business Intelligence & Reporting Department** to overcome strict privacy regulations that prohibit the use of production customer data for development and testing.

The generator produces a coherent, relational dataset (OLTP) capable of feeding an end-to-end Business Intelligence workflow—from data generation and ETL (Extract, Transform, Load) to Data Warehousing and DataMart creation.

## 🚀 Key Features
* **Privacy Compliance:** Generates fully synthetic data that mirrors real-world consumer behavior without exposing sensitive information.
* **Relational Integrity:** Produces **10 interrelated tables** with consistent surrogate keys, ensuring valid joins across Customers, Accounts, Transactions, and Merchants.
* **Behavioral Segmentation:** Simulates distinct customer personas (e.g., *Blue Collar, Young Professionals, Students*) with unique income cycles, spending habits, and risk profiles.
* **Temporal Realism:** Incorporates salary days, rent cycles, bill due dates, seasonal spending spikes, and weekend/holiday effects.
* **Advanced Logic:** Includes "Discipline Factors" for budgeting simulation and a "Cash Shadow" mechanism to mimic untraceable cash usage after ATM withdrawals.
* **Reproducibility:** Uses seeded randomization to ensure the same dataset can be regenerated consistently for testing pipelines.

## 📂 Dataset Structure
The generator creates a normalized schema consisting of the following 10 tables:

| Table Name | Description |
| :--- | :--- |
| **Customers** | Demographics, credit scores, income levels, and behavioral segments. |
| **Accounts** | Account balances, overdraft limits, and account types. |
| **Transactions** | The core fact table containing amounts, timestamps, and balance updates. |
| **Merchants** | Merchant IDs, names, and location data. |
| **Categories** | Transaction categories (e.g., Grocery, Rent, Bills) and essential/non-essential flags. |
| **Channels** | Transaction channels (ATM, POS, Mobile, Branch). |
| **Calendar** | Date dimension with holiday flags, fiscal quarters, and weekend indicators. |
| **Branches** | Bank branch locations and identifiers. |
| **Cities** | Geographic data for branches and merchants. |
| **Transaction Types** | Definitions for transfers, payments, and deposits. |

## 🧠 Generation Logic & Mathematics
The core of this project lies in its ability to simulate human behavior mathematically rather than just randomization.

## 🚀 Project Features

### 👤 Customer Modeling
- Demographics (age, gender, education, employment)
- Segmentation (students, young professionals, families, retirees, high-income)
- Income composition:
  - Salary
  - Passive income
  - Rental income
- Credit score & dependents
- Behavioral weights:
  - Category preferences
  - Channel preferences
  - Time-of-day activity
  - Weekly transaction intensity

### 💳 Transaction Generation
- One full year of transactions per customer
- Deterministic recurring events:
  - Salary payments (fixed monthly day)
  - Bills & utilities (fixed due dates)
  - Rent payments (in/out)
- Probabilistic daily spending:
  - Grocery, gas, restaurant, shopping, healthcare, entertainment, etc.
- Channel modeling:
  - POS, Online, ATM, Branch
- Balance tracking:
  - Previous balance
  - Balance after transaction
- Currency: **USD**

### 1. Customer Generation
Customers are generated using **segment-specific distributions**:

- Income scaling
- Employment patterns
- Education levels
- Channel & category weights
- Transaction frequency

### 2. Transaction Logic

Each transaction follows:

- **Date**  
  - Uniform weekly distribution  
  - Normal distribution by hour (`μ ≈ evening`, `σ` per segment)

- **Category**
  - Weighted by customer preferences

- **Channel**
  - Weighted by digital vs physical behavior

- **Amount**
  - Category-specific ranges
  - Scaled by income level

- **Balance Update**
  - Credits increase balance
  - Debits decrease balance


  ### 2.1 The "Discipline" Factor
  A "daily discretionary cap" is calculated for each customer based on their income vs. fixed outflows (rent/bills). A sigmoidal function is applied to simulate "budget discipline," determining if a customer saves money or overspends into their overdraft.

  ### 2.2 The "Cash Shadow"
  To solve the issue of ATM withdrawals simply being "money exiting the system," this project implements a **Cash Shadow**. When a customer withdraws cash, a "shadow balance" is created. This temporarily suppresses visible card spending (POS transactions) until the cash is "spent" virtually, mimicking real-world cash substitution.

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy (for vectorization and sampling), Faker
* **Environment:** Dockerized setup (compatible with PostgreSQL and Spark)

## 📊 End-to-End Workflow Context
While this repository focuses on the **Data Generation** phase, it was designed as the first step in a larger pipeline:
1.  **Generation:** Python scripts produce CSVs/DataFrames.
2.  **OLTP Storage:** Data is loaded into **PostgreSQL** (Normalized Schema).
3.  **ETL:** **Apache Spark** extracts and transforms data (SCD Type 1 & 3).
4.  **OLAP Storage:** Data is loaded into **ClickHouse** (Star Schema).
5.  **Analytics:** DataMarts are built for reporting.

## 🔧 Usage
1.  Clone the repository:
    ```bash
    git clone [https://github.com/AlperCamli/Synthetic-Banking-Transaction-Data-Generator.git](https://github.com/AlperCamli/Synthetic-Banking-Transaction-Data-Generator.git)
    ```
2.  Install dependencies:
    ```bash
    pip install pandas numpy
    ```
3.  Run the generator:
    ```bash
    python main.py
    ```
    *(Note: Check configuration files to adjust the number of customers or date range).*

## 📜 License
This project is open-source and available under the MIT License.



---

## 🗂 Repository Structure
```bash
Synthetic-Banking-Transaction-Data-Generator/
│
├── Analytics/
│   ├── Customer/
│   │   ├── age_distribution.py                    # Analysis of age demographics per segment
│   │   ├── customer_data_visualization.py         # General visualization logic for customer attributes
│   │   ├── demographics.py                        # Core demographic statistical checks
│   │   ├── salary_age_by_segmet.py                # Correlation analysis between age and income
│   │   └── total_monthly_income distribution.py   # Income distribution checks across the population
│   │
│   └── Transactions/
│       ├── balance_movement_in_time.py            # Temporal analysis of account balances
│       └── transaction_analytics.py               # General transaction volume and pattern analytics
│
├── Data/                                          # Storage for generated CSV outputs
│
├── Data_Generators/
│   ├── Old Versions/
│   │   ├── transaction_generator_old_v1.py
│   │   ├── transaction_generator_old_v2.py
│   │   └── transaction_generator_old_v3.py
│   ├── customer_data.py                           # Logic for generating customer profiles & segments
│   └── transaction_data_generator.py              # Logic for generating transactions & cash shadow
│
├── utils/
│   └── Segment_normalization.py                   # Helper functions for normalizing segment weights
│
└── README.md
```


---

## ⚙️ How It Works

### 1️⃣ Customer Generation
Customers are generated using **segment-specific distributions**:

- Income scaling
- Employment patterns
- Education levels
- Channel & category weights
- Transaction frequency

### 2️⃣ Transaction Logic

Each transaction follows:

- **Date**  
  - Uniform weekly distribution  
  - Normal distribution by hour (`μ ≈ evening`, `σ` per segment)

- **Category**
  - Weighted by customer preferences

- **Channel**
  - Weighted by digital vs physical behavior

- **Amount**
  - Category-specific ranges
  - Scaled by income level

- **Balance Update**
  - Credits increase balance
  - Debits decrease balance

---

## ▶️ Usage

### Install dependencies
```bash
pip install -r requirements.txt
Generate customers
bash
Copy code
python src/generate_customers.py
Generate transactions
bash
Copy code
python src/generate_transactions.py
Output:

bash
Copy code
data/generated/transactions.csv
📊 Example Analyses
Included notebooks demonstrate:

Age distribution

Income distribution (weighted & unweighted)

Transaction volume by segment

Channel usage patterns

Balance evolution over time

🎯 Use Cases
BI dashboards (Power BI / Tableau)

SQL analytics practice

Spark / ETL pipelines

Fraud detection prototyping

Data warehouse modeling

ML feature engineering

📜 License
MIT License — free to use for academic and commercial purposes.

yaml
Copy code

---

# 📄 requirements.txt

pandas
numpy
matplotlib
seaborn

yaml
Copy code

---

# 📄 .gitignore

pycache/
.ipynb_checkpoints/
.env
data/generated/

markdown
Copy code

---

# 🏆 Final Recommendation

This project is **absolutely portfolio-ready**.

If you want, next we can:
- Add **ClickHouse / PostgreSQL schema**
- Add **dbt models**
- Add **Spark ETL**
- Add **data quality checks**
- Write a **project story for LinkedIn / CV**

Just tell me what the next showcase goal is.
