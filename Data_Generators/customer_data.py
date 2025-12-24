import pandas as pd
import numpy as np
from datetime import date, timedelta
import math


# ============================================================
# CONFIG — tweak these freely
# ============================================================

SEED = 123
N_CUSTOMERS = 100
BASELINE_MONTHLY_USD = 500.0
TODAY = date.today()  # used for DOB generation

# Segment proportions (sum to ~1.0)
SEGMENT_PROPS = {
    "young_students": 0.17,
    "young_professionals": 0.16,
    "blue_collar": 0.24,
    "mid_career_families": 0.21,
    "retirees": 0.14,
    "high_income": 0.06,
}

# Per-customer noise on channel/category RANDOM weights
NOISE = 0.05  # e.g., 0.05 = ±5%

# Dependents effect on weekly txns:
# weekly_transaction_rate = base_weekly + DEPENDENT_TRANSACTION * num_dependents
DEPENDENT_TRANSACTION = 2

# Probability someone is responsible for fixed monthly bills (constant day)
P_BILL_RESPONSIBLE = {
    "young_students": 0.1,
    "young_professionals": 0.6,
    "blue_collar": 0.85,
    "mid_career_families": 0.5,
    "retirees": 0.90,
    "high_income": 0.95,
}

# Real estate ownership probabilities (not applied to students or blue_collar)
P_REAL_ESTATE = {
    "young_professionals": 0.02,
    "mid_career_families": 0.1,
    "retirees": 0.40,
    "high_income": 0.50,
}

# Passive investment income probabilities (independent of RE except retirees)
P_PASSIVE_INCOME = {
    "young_students": 0.01,
    "young_professionals": 0.05,
    "blue_collar": 0.02,
    "mid_career_families": 0.20,
    "retirees": 0.2,
    "high_income": 0.80,
}

# Monthly passive income ranges (USD) for segments that can have it ADJUST THE
PASSIVE_INCOME_MONTHLY_USD_RANGE = {
    "young_professionals": (0, 200),
    "mid_career_families": (200, 600),
    "retirees": (300, 700),     # only if RE=True
    "high_income": (2000, 5000)
}

# Monthly rent IN (rental income) ranges when real estate is owned
RENT_IN_MONTHLY_USD_RANGE = {
    "young_professionals": (150, 600),
    "mid_career_families": (250, 1200),
    "retirees": (200, 600),
    "high_income": (1000, 4000),
}

# Monthly rent OUT (expense) as % of salary for renters (per segment)
RENT_OUT_PCT_RANGE = {
    "young_students": (0.5, 0.6),
    "young_professionals": (0.3, 0.5),
    "blue_collar": (0.3, 0.5),
    "mid_career_families": (0.15, 0.25),
    "retirees": (0.30, 0.40),
    "high_income": (0.10, 0.25),
}

# Constant monthly date windows
INCOME_FIRST_WEEK_RANGE = (1, 6)
INCOME_LAST_WEEK_RANGE  = (25, 28)
INCOME_FIRST_WEEK_WEIGHT = 0.70

BILLS_DAY_RANGE = (2, 15)   # constant per bill-responsible person
RENT_DAY_RANGE  = (1, 10)   # constant per renter

# Channels (weights columns will be w_channel_<name>)
CHANNELS = ["POS", "online", "ATM", "branch"]

# ALL categories (for documentation); constant-payment ones excluded from random weights
ALL_CATEGORIES = [
    "gas", "grocery", "restaurant", "shopping", "healthcare",
    "transportation", "public_transportation", "income", "bill_and_payments",
    "entertainment", "ATM", "rent",
]

# Random categories (used for w_cat_* columns)
RANDOM_CATEGORIES = [
    "gas", "grocery", "restaurant", "shopping", "healthcare",
    "transportation", "public_transportation", "entertainment", "ATM",
]

# ============================================================
# SEGMENT DEFINITIONS (random category weights only)
# ============================================================

SEGMENTS = {
    "young_students": {
        "age_range": (18, 25),
        "marital_p":   {"single": 0.95, "married": 0.05},
        "education_p": {"university": 0.70, "highschool": 0.30},
        "employment_p": {"student": 0.60, "employed": 0.20, "part_time": 0.10, "unemployed": 0.10},
        "job_titles": ["University Student", "Intern", "Assistant", "Barista"],
        "housing_p": {"rent": 0.60, "family": 0.40},
        "month_factor_range": (0.5, 1.2),
        "credit_mu": 625, "credit_sigma": 30,
        "num_dep_range": (0, 0.5),
        "weekly_txn_rate": (10, 16),
        "channel_shares": {"POS": 0.70, "online": 0.299, "ATM": 0.001, "branch": 0.000},
        "tod_mean_std": (17.5, 4.0),
        "category_weights": {'gas': 0.02, 'grocery': 0.08, 'restaurant': 0.599, 'shopping': 0.1,
                             'healthcare': 0.02, 'transportation': 0.03, 'public_transportation': 0.0,
                             'entertainment': 0.15, 'ATM': 0.001},
    },
    "young_professionals": {
        "age_range": (26, 34),
        "marital_p":   {"single": 0.70, "married": 0.30},
        "education_p": {"bachelor": 0.60, "master": 0.20, "highschool": 0.20},
        "employment_p": {"employed": 1.0},
        "job_titles": ["Software Engineer", "Marketing Specialist", "Analyst", "Designer"],
        "housing_p": {"rent": 0.85, "own": 0.05, "family": 0.1},
        "month_factor_range": (1, 2.5),
        "credit_mu": 700, "credit_sigma": 35,
        "num_dep_range": (0, 0.8),
        "weekly_txn_rate": (12, 17),
        "channel_shares": {"POS": 0.678, "online": 0.32, "ATM": 0.002, "branch": 0.000},
        "tod_mean_std": (18.0, 3.5),
        "category_weights": {'gas': 0.078, 'grocery': 0.156, 'restaurant': 0.391, 'shopping': 0.234,
                             'healthcare': 0.006, 'transportation': 0.016, 'public_transportation': 0.0,
                             'entertainment': 0.117, 'ATM': 0.002},
    },
    "blue_collar": {
        "age_range": (18, 45),
        "marital_p":   {"single": 0.45, "married": 0.55},
        "education_p": {"primary": 0.25, "secondary": 0.45, "highschool": 0.30},
        "employment_p": {"employed": 1.0},
        "job_titles": ["Technician", "Driver", "Shopkeeper", "Operator"],
        "housing_p": {"rent": 0.80, "own": 0.05, "family": 0.15},
        "month_factor_range": (1.0, 1.5),
        "credit_mu": 655, "credit_sigma": 30,
        "num_dep_range": (0, 3),
        "weekly_txn_rate": (8, 14),
        "channel_shares": {"POS": 0.68, "online": 0.239, "ATM": 0.02, "branch": 0.001},
        "tod_mean_std": (17.5, 4.0),
        "category_weights": {'gas': 0.106, 'grocery': 0.300, 'restaurant': 0.17, 'shopping': 0.129,
                             'healthcare': 0.165, 'transportation': 0.003, 'public_transportation': 0.0,
                             'entertainment': 0.007, 'ATM': 0.070},
    },
    "mid_career_families": {
        "age_range": (35, 55),
        "marital_p":   {"married": 0.95, "single": 0.05},
        "education_p": {"bachelor": 0.60, "master": 0.15, "highschool": 0.20},
        "employment_p": {"employed": 1.0},
        "job_titles": ["Project Manager", "Accountant", "Sales Manager", "Teacher"],
        "housing_p": {"own": 0.65, "rent": 0.35},
        "month_factor_range": (2.0, 5.0),
        "credit_mu": 725, "credit_sigma": 30,
        "num_dep_range": (1, 3),
        "weekly_txn_rate": (12, 18),
        "channel_shares": {"POS": 0.679, "online": 0.36, "ATM": 0.02, "branch": 0.001},
        "tod_mean_std": (17.0, 4.0),
        "category_weights": {'gas': 0.209, 'grocery': 0.249, 'restaurant': 0.144, 'shopping': 0.250,
                             'healthcare': 0.072, 'transportation': 0.004, 'public_transportation': 0.0,
                             'entertainment': 0.072, 'ATM': 0.00},
    },
    "retirees": {
        "age_range": (55, 80),
        "marital_p":   {"married": 0.60, "widowed": 0.25, "single": 0.15},
        "education_p": {"highschool": 0.50, "secondary": 0.30, "bachelor": 0.20},
        "employment_p": {"retired": 1.0},
        "job_titles": ["Retired"],
        "housing_p": {"own": 0.75, "family": 0.10, "rent": 0.15},
        "month_factor_range": (0.7, 2.5),  # actual pick uses RE flag below
        "credit_mu": 685, "credit_sigma": 30,
        "num_dep_range": (0, 0.8),
        "weekly_txn_rate": (4, 8),
        "channel_shares": {"POS": 0.55, "online": 0.35, "ATM": 0.08, "branch": 0.02},
        "tod_mean_std": (15.5, 4.5),
        "category_weights": {'gas': 0.277, 'grocery': 0.414, 'restaurant': 0.063, 'shopping': 0.063,
                             'healthcare': 0.05, 'transportation': 0.003, 'public_transportation': 0.0,
                             'entertainment': 0.005, 'ATM': 0.125},
    },
    "high_income": {
        "age_range": (35, 60),
        "marital_p":   {"married": 0.70, "single": 0.30},
        "education_p": {"master": 0.40, "phd": 0.10, "bachelor": 0.50},
        "employment_p": {"employed": 1.0},
        "job_titles": ["Senior Manager", "Director", "CTO", "CFO"],
        "housing_p": {"own": 0.95, "rent": 0.05},
        "month_factor_range": (5.0, 10.0),
        "credit_mu": 785, "credit_sigma": 30,
        "num_dep_range": (0, 2.5),
        "weekly_txn_rate": (16, 25),
        "channel_shares": {"POS": 0.445, "online": 0.50, "ATM": 0.005, "branch": 0.05},
        "tod_mean_std": (17.5, 3.5),
        "category_weights": {'gas': 0.095, 'grocery': 0.095, 'restaurant': 0.569, 'shopping': 0.19,
                             'healthcare': 0.006, 'transportation': 0.013, 'public_transportation': 0.0,
                             'entertainment': 0.032, 'ATM': 0.0},
    },
}

# ============================================================
# Helpers
# ============================================================

rng = np.random.default_rng(SEED)

first_names_m = ["Ahmet", "Mehmet", "Ali", "Mustafa", "Emre", "Can", "Oğuz", "Burak", "Kerem", "Uğur"]
first_names_f = ["Ayşe", "Fatma", "Elif", "Zeynep", "Merve", "Ece", "Seda", "Melis", "Derya", "Buse"]
last_names = ["Yılmaz", "Kaya", "Demir", "Çelik", "Şahin", "Yıldız", "Yıldırım", "Aydın", "Öztürk", "Arslan", "Candan", "Kara", "Topal", "Ak", "Kaplan", "Er", "Engin", "Balcı", "Bolat", "Güler"]

def pick_name(gender: str):
    fn = rng.choice(first_names_f if gender == "female" else first_names_m)
    ln = rng.choice(last_names)
    return fn, ln

"""def dob_for_age_range(min_age: int, max_age: int) -> str:
    age = int(rng.integers(min_age, max_age + 1))
    day_offset = int(rng.integers(0, 365))
    return (TODAY - timedelta(days=age * 365 + day_offset)).isoformat()"""

def dob_for_age_range(min_age: int, max_age: int, std_factor: float) -> str:
    avg_age = (min_age + max_age) * 0.5
    range_age = max_age - min_age
    std_age = range_age * std_factor
    age = int(rng.normal(avg_age, std_age))
    while age < min_age * 0.75 or age < 18:
        age = int(rng.normal(avg_age, std_age))
    day_offset = int(rng.integers(0, 365))
    return (TODAY - timedelta(days=age * 365 + day_offset)).isoformat()

def salary_day() -> int:
    if rng.random() < INCOME_FIRST_WEEK_WEIGHT:
        return int(rng.integers(INCOME_FIRST_WEEK_RANGE[0], INCOME_FIRST_WEEK_RANGE[1] + 1))
    return int(rng.integers(INCOME_LAST_WEEK_RANGE[0], INCOME_LAST_WEEK_RANGE[1] + 1))

def bills_due_day() -> int:
    return int(rng.integers(BILLS_DAY_RANGE[0], BILLS_DAY_RANGE[1] + 1))

def rent_due_day() -> int:
    return int(rng.integers(RENT_DAY_RANGE[0], RENT_DAY_RANGE[1] + 1))

def clip_credit(score: float) -> int:
    return int(max(300, min(850, round(score))))

def add_noise(value: float, strength: float = NOISE) -> float:
    return max(0.0, value * (1.0 + rng.normal(0.0, strength)))

def normalize(d: dict) -> dict:
    s = float(sum(d.values()))
    if s == 0:
        k = len(d)
        return {key: 1.0 / k for key in d}
    return {k: v / s for k, v in d.items()}

def weighted_choice(d: dict) -> str:
    keys = list(d.keys())
    vals = np.array(list(d.values()), dtype=float)
    vals = vals / vals.sum()
    return str(rng.choice(keys, p=vals))

# ============================================================
# Generation
# ============================================================

def main():
    segments = list(SEGMENT_PROPS.keys())
    seg_probs = np.array(list(SEGMENT_PROPS.values()), dtype=float)
    seg_probs = seg_probs / seg_probs.sum()

    rows = []
    for i in range(1, N_CUSTOMERS + 1):
        seg_name = str(rng.choice(segments, p=seg_probs))
        seg = SEGMENTS[seg_name]

        # Demographics
        gender = str(rng.choice(["male", "female"]))
        first_name, last_name = pick_name(gender)

        if seg_name == "retirees":
            dob = dob_for_age_range(*seg["age_range"], 0.4)
        elif seg_name in ("mid_career_families", "young_professionals"):
            dob = dob_for_age_range(*seg["age_range"], 0.4)
        elif seg_name in ("high_income"):
            dob = dob_for_age_range(*seg["age_range"], 0.3)
        else:
            dob = dob = dob_for_age_range(*seg["age_range"], 0.7)

        marital = weighted_choice(seg["marital_p"])
        education = weighted_choice(seg["education_p"])
        employment = weighted_choice(seg["employment_p"])
        job_title = str(rng.choice(seg["job_titles"]))
        housing = weighted_choice(seg["housing_p"])

        # ---- Income components (monthly) ----
        # Salary (earned) from baseline scaling

        if seg_name in ("retirees"):
            lo, hi = seg["month_factor_range"]
            avg_salary = (lo + hi) * 0.5
            range_salary = hi - lo
            std_salary = range_salary * 0.25
            monthly_factor_salary = float(rng.normal(avg_salary, std_salary))
            monthly_salary_usd = monthly_factor_salary * BASELINE_MONTHLY_USD

            # Real estate (optional)
            has_re = bool(rng.random() < P_REAL_ESTATE.get(seg_name, 0.0))
            has_real_estate_income = has_re
            monthly_rent_in_usd = 0.0
            if has_re:
                rlo, rhi = RENT_IN_MONTHLY_USD_RANGE[seg_name]
                avg_rent_in = (rlo + rhi) * 0.5
                range_rent_in = rhi - rlo
                std_rent_in = range_rent_in * 0.1
                monthly_rent_in_usd = float(rng.normal(avg_rent_in, std_rent_in))

            # Passive non-rental (optional)
            p_passive = P_PASSIVE_INCOME.get(seg_name, 0.0)
            has_passive_income = bool(rng.random() < p_passive)
            monthly_passive_income_usd = 0.0
            if has_passive_income:
                plo, phi = PASSIVE_INCOME_MONTHLY_USD_RANGE[seg_name]
                avg_passive_income = (plo + phi) * 0.5
                range_passive_income = hi - lo
                std_passive_income = range_passive_income * 0.1
                monthly_passive_income_usd = float(rng.normal(avg_passive_income, std_passive_income))
        elif seg_name in ("mid_career_families", "young_professionals"):
            lo, hi = seg["month_factor_range"]
            avg_salary = (lo + hi) * 0.5
            range_salary = hi - lo
            std_salary = range_salary * 0.3
            monthly_factor_salary = float(rng.normal(avg_salary, std_salary))
            monthly_salary_usd = monthly_factor_salary * BASELINE_MONTHLY_USD

            # Real estate (optional)
            has_re = bool(rng.random() < P_REAL_ESTATE.get(seg_name, 0.0))
            has_real_estate_income = has_re
            monthly_rent_in_usd = 0.0
            if has_re:
                rlo, rhi = RENT_IN_MONTHLY_USD_RANGE[seg_name]
                avg_rent_in = (rlo + rhi) * 0.5
                range_rent_in = rhi - rlo
                std_rent_in = range_rent_in * 0.1
                monthly_rent_in_usd = float(rng.normal(avg_rent_in, std_rent_in))

            # Passive non-rental (optional)
            p_passive = P_PASSIVE_INCOME.get(seg_name, 0.0)
            has_passive_income = bool(rng.random() < p_passive)
            monthly_passive_income_usd = 0.0
            if has_passive_income:
                plo, phi = PASSIVE_INCOME_MONTHLY_USD_RANGE[seg_name]
                avg_passive_income = (plo + phi) * 0.5
                range_passive_income = phi - plo
                std_passive_income = range_passive_income * 0.1
                monthly_passive_income_usd = float(rng.normal(avg_passive_income, std_passive_income))
        elif seg_name in ("high_income"):
            lo, hi = seg["month_factor_range"]
            avg_salary = (lo + hi) * 0.5
            range_salary = hi - lo
            std_salary = range_salary * 0.4
            monthly_factor_salary = float(rng.normal(avg_salary, std_salary))
            monthly_salary_usd = monthly_factor_salary * BASELINE_MONTHLY_USD

            # Real estate (optional)
            has_re = bool(rng.random() < P_REAL_ESTATE.get(seg_name, 0.0))
            has_real_estate_income = has_re
            monthly_rent_in_usd = 0.0
            if has_re:
                rlo, rhi = RENT_IN_MONTHLY_USD_RANGE[seg_name]
                avg_rent_in = (rlo + rhi) * 0.5
                range_rent_in = rhi - rlo
                std_rent_in = range_rent_in * 0.1
                monthly_rent_in_usd = float(rng.normal(avg_rent_in, std_rent_in))

            # Passive non-rental (optional)
            p_passive = P_PASSIVE_INCOME.get(seg_name, 0.0)
            has_passive_income = bool(rng.random() < p_passive)
            monthly_passive_income_usd = 0.0
            if has_passive_income:
                plo, phi = PASSIVE_INCOME_MONTHLY_USD_RANGE[seg_name]
                avg_passive_income = (plo + phi) * 0.5
                range_passive_income = phi - plo
                std_passive_income = range_passive_income * 0.1
                monthly_passive_income_usd = float(rng.normal(avg_passive_income, std_passive_income))

        else:
            # young_students, blue_collar
            lo, hi = seg["month_factor_range"]
            avg_salary = (lo + hi) * 0.5
            range_salary = hi - lo
            std_salary = range_salary * 0.4
            monthly_factor_salary = float(rng.normal(avg_salary, std_salary))
            monthly_salary_usd = monthly_factor_salary * BASELINE_MONTHLY_USD
            has_real_estate_income = False
            has_passive_income = False
            monthly_passive_income_usd = 0.0
            monthly_rent_in_usd = 0.0

        # Rent OUT (expense) if renter, as % of salary
        is_renter = (housing == "rent")
        monthly_rent_out_usd = 0.0
        if is_renter:
            p_lo, p_hi = RENT_OUT_PCT_RANGE[seg_name]
            avg_rent_out = (p_lo + p_hi) * 0.5
            range_rent_out = p_hi - p_lo
            std_rent_out = range_rent_out * 0.1
            pct = float(rng.normal(avg_rent_out, std_rent_out))
            monthly_rent_out_usd = round(monthly_salary_usd * pct, 2)

        # Annual income (gross inflow): salary + passive + rent_in
        annual_income = round((monthly_salary_usd + monthly_passive_income_usd + monthly_rent_in_usd) * 12.0, 2)

        # ---- Constant monthly dates ----
        income_day = salary_day()
        bill_resp = bool(rng.random() < P_BILL_RESPONSIBLE[seg_name])
        bills_day = bills_due_day() if bill_resp else None
        rent_day_const = rent_due_day() if is_renter else None

        # ---- Credit, dependents, behavior ----
        credit = clip_credit(rng.normal(seg["credit_mu"], seg["credit_sigma"]))
        dep_lo, dep_hi = seg["num_dep_range"]
        num_dep = max(math.floor(dep_lo), int(round(rng.normal((dep_lo + dep_hi) * 0.5, 0.2))))

        wk_lo, wk_hi = seg["weekly_txn_rate"]
        base_weekly = max(wk_lo, int(round(rng.normal((wk_lo + wk_hi) * 0.5, 2))))
        weekly_transaction_rate = int(base_weekly + DEPENDENT_TRANSACTION * num_dep)

        tod_mean, tod_std = seg["tod_mean_std"]
        channels = normalize({k: add_noise(v) for k, v in seg["channel_shares"].items()})
        cat_weights = normalize({k: add_noise(seg["category_weights"][k]) for k in RANDOM_CATEGORIES})

        row = {
            # core
            "customer_id": f"CUST{i:04d}",
            "first_name": first_name,
            "last_name": last_name,
            "date_of_birth": dob,
            "gender": gender,
            "marital_status": marital,
            "education_level": education,
            "employment_status": employment,
            "job_title": job_title,
            "housing": housing,

            # income components (monthly) + annual gross inflow
            "monthly_salary_usd": round(monthly_salary_usd, 2),
            "monthly_passive_income_usd": round(monthly_passive_income_usd, 2),
            "monthly_rent_in_usd": round(monthly_rent_in_usd, 2),
            "monthly_rent_out_usd": round(monthly_rent_out_usd, 2),
            "annual_income": annual_income,

            # monthly constants
            "monthly_salary_day": income_day,
            "bill_responsible": bill_resp,
            "bills_due_day": bills_day,
            "is_renter": is_renter,
            "rent_due_day": rent_day_const,

            # credit/dependents
            "credit_score": int(credit),
            "num_dependents": num_dep,

            # flags
            "segment": seg_name,
            "has_passive_income": has_passive_income,
            "has_real_estate_income": has_real_estate_income,

            # behavior knobs
            "weekly_txn_rate": weekly_transaction_rate,
            "tod_mean_hour": round(tod_mean, 2),
            "tod_std_hour": round(tod_std, 2),
        }

        # channel shares
        for ch in CHANNELS:
            row[f"w_channel_{ch}"] = round(channels[ch], 4)

        # random category weights only
        for cat in RANDOM_CATEGORIES:
            row[f"w_cat_{cat}"] = round(cat_weights[cat], 4)

        rows.append(row)

    customers = pd.DataFrame(rows)
    customers.to_csv("customers.csv", index=False)
    print(f"✅ Generated customers.csv with {len(customers)} rows")

if __name__ == "__main__":
    main()
