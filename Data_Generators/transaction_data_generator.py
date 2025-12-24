import math
import random
import string
from dataclasses import dataclass
from datetime import date, datetime, timedelta, time
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# =========================
# Config & constants
# =========================

@dataclass
class DEFAULTS:
    start_date: str = "2024-01-01"
    end_date: str = "2024-12-31"
    currency: str = "USD"

    # Fixed posting times so salary lands first
    salary_time: time = time(9, 0, 0)
    bills_time: time = time(10, 0, 0)
    rent_time: time = time(10, 30, 0)
    passive_time: time = time(11, 0, 0)
    rent_in_time: time = time(9, 30, 0)

    # Balance pressure: when balance < 20% (hi) or 10% (lo) of salary
    thin_threshold_hi: float = 0.30
    thin_threshold_lo: float = 0.10
    # Frequency thinning (counts)
    thin_factor_hi: float = 0.50
    thin_factor_lo: float = 0.35
    # Amount shrinking (spend pressure on sizes)
    amount_pressure_hi: float = 0.95
    amount_pressure_lo: float = 0.85

    # ATM "cash shadow": after ATM wd, card-visible spend drops and recovers over days
    cash_shadow_decay: float = 0.55         # decay each day (0.75 → 25% recovers daily)
    cash_shadow_cap: float = 0.90           # cap on shadow
    cash_shadow_gain_div_salary: float = 1.00  # scale ATM gain vs salary
    cash_shadow_amount_weight: float = 0.20  # portion of shadow that hits amount sizes

    # Random
    random_state: int = 45


# Transaction type IDs (your mapping confirmed)
TT = {
    "purchase": 1,         # debit
    "atm_withdrawal": 2,   # debit
    "bill_payment": 3,     # debit
    "salary": 4,           # credit
    "passive_income": 5,   # credit
    "rent_out": 6,         # debit
    "rent_in": 7,          # credit
    "transfer_out": 8,     # debit
    "transfer_in": 9,      # credit
}

CREDIT_TYPES = {TT["salary"], TT["passive_income"], TT["rent_in"], TT["transfer_in"]}
DEBIT_TYPES  = {TT["purchase"], TT["atm_withdrawal"], TT["bill_payment"], TT["rent_out"], TT["transfer_out"]}

# Default (fallback) category ranges if segment not found
DEFAULT_CATEGORY_RANGES = {
    "gas": (20, 120),
    "grocery": (25, 180),
    "restaurant": (10, 90),
    "shopping": (15, 300),
    "healthcare": (20, 600),
    "transportation": (5, 45),     # not public transportation
    "entertainment": (8, 140),
    "atm": (50, 400),              # ATM withdrawal
}

# Map education text → numeric level (1..6)
EDU_LEVELS = {
    "primary": 1, "elementary": 1,
    "middle_school": 2, "secondary": 2,
    "highschool": 3, "high school": 3, "lycee": 3, "vocational": 3,
    "university": 4, "college": 4, "bachelor": 4, "undergraduate": 4,
    "master": 5, "msc": 5, "mba": 5,
    "phd": 6, "doctorate": 6,
}



# Baseline discipline per level (higher education → higher baseline)
# Use these as *inputs* to the age+edu blend below.
EDU_BASELINE_DISCIPLINE = {
    1: 0.00,  # primary / elementary
    2: 0.20,  # middle / secondary from 50 to 45
    3: 0.25,  # highschool / vocational
    4: 0.30,  # university / bachelor
    5: 0.30,  # master / MBA
    6: 0.40,  # PhD / doctorate
}
# Put near your configs
SEG_DISC_MULTIPLIER = {
    "young_students":         1.10,  # more, smaller purchases
    "young_professionals":    1.10,
    "high_income":       0.90,
    "mid_career_families":    0.85,  # fewer, larger purchases
    "retirees":               1.00,
}


# Segment-specific ranges (tune as you like)
SEGMENT_CATEGORY_RANGES: Dict[str, Dict[str, Tuple[float, float]]] = {
    "young_students": {
        "gas": (0, 40),
        "grocery": (10, 30),
        "restaurant": (8, 30),
        "shopping": (10, 120),
        "healthcare": (10, 30),
        "transportation": (15, 150),
        "entertainment": (7, 80),
        "atm": (10, 50),
    },
    "mid_career_families": {
        "gas": (25, 60),
        "grocery": (30, 100),
        "restaurant": (40, 90),
        "shopping": (20, 300),
        "healthcare": (25, 200),
        "transportation": (45, 400),
        "entertainment": (10, 180),
        "atm": (60, 100),
    },
    "young_professionals": {
        "gas": (20, 40),
        "grocery": (15, 50),
        "restaurant": (15, 50),
        "shopping": (30, 200),
        "healthcare": (30, 30),
        "transportation": (10, 80),
        "entertainment": (8, 120),
        "atm": (60, 100),
    },
    "retirees": {
        "gas": (40, 60),
        "grocery": (20, 40),
        "restaurant": (20, 30),
        "shopping": (10, 30),
        "healthcare": (40, 200),
        "transportation": (15, 35),
        "entertainment": (8, 20),
        "atm": (40, 200),
    },
    "blue_collar": {
        "gas": (10, 20),
        "grocery": (20, 50),
        "restaurant": (8, 30),
        "shopping": (10, 140),
        "healthcare": (15, 100),
        "transportation": (5, 30),
        "entertainment": (8, 20),
        "atm": (30, 100),
    },
    "high_income": {
        "gas": (30, 80),
        "grocery": (40, 150),
        "restaurant": (20, 80),
        "shopping": (50, 500),
        "healthcare": (15, 200),
        "transportation": (20, 200),
        "entertainment": (8, 300),
        "atm": (100, 300),
    }
}

# Column-name maps for weights
WEIGHT_COL_TO_CATEGORY = {
    "gas": "gas",
    "grocery": "grocery",
    "restaurant": "restaurant",
    "shopping": "shopping",
    "healthcare": "healthcare",
    "transportation": "transportation",
    "entertainment": "entertainment",
    "ATM": "atm",  # we treat ATM as a "category" to pick ATM withdrawals
}
CHANNEL_COLUMNS = {"POS": "pos", "online": "online", "ATM": "atm", "branch": "branch"}


# =========================
# Helpers
# =========================

from collections import defaultdict


from dataclasses import dataclass
from datetime import date, timedelta
import numpy as np


def split_weekly_rate(weekly_rate: float, ch_weights: dict) -> tuple[float, float]:#burayla oyna
    """
    Split customer's weekly_txn_rate into (ATM part, Purchase part) using the channel mix.
    This prevents purchase caps from suppressing ATM frequency.
    """
    atm_share = float(ch_weights.get("atm", 0.0))
    atm_share = float(np.clip(atm_share, 0.0, 0.85))   # keep at least some purchase activity
    purch_share = max(0.05, 1.0 - atm_share)
    total = atm_share + purch_share
    atm_share /= total; purch_share /= total
    return weekly_rate * atm_share, weekly_rate * purch_share

def purchase_count_multiplier(balance: float, salary: float, day: date,
                              cash_shadow: float, discipline: float, cfg) -> float:
    """
    Light-touch adjustments (no hard clamp to near-zero).
    Counts mostly follow weekly_txn_rate; balance/cash shadow nudge slightly.
    """
    thr_hi = salary * cfg.thin_threshold_hi
    thr_lo = salary * cfg.thin_threshold_lo
    base = 1.00
    if balance < thr_lo:   base = 0.65
    elif balance < thr_hi: base = 0.78

    weekend = 1.06 if day.weekday() >= 5 else 1.00
    dom     = 1.03 if day.day <= 5 else (0.98 if day.day >= 27 else 1.00)
    cash    = max(0.55, 1.0 - 0.60 * float(np.clip(cash_shadow, 0.0, 1.0)))
    noise   = float(np.clip(np.random.normal(1.0, 0.10), 0.80, 1.25))
    mult    = base * weekend * dom * cash * noise
    return float(np.clip(mult, 0.5, 1.4))

def atm_count_multiplier(balance: float, salary: float, day: date,
                         cash_shadow: float, discipline: float, crashed_last_q: bool) -> float:
    """
    ATM becomes more frequent with cash usage and low discipline; slightly reduced after a crash.
    """
    base = 1.0 + 0.6 * float(np.clip(cash_shadow, 0.0, 1.0)) + 0.3 * (1.0 - float(np.clip(discipline, 0.0, 1.0)))
    if crashed_last_q:
        base *= 0.85
    noise = float(np.clip(np.random.normal(1.0, 0.15), 0.7, 1.5))
    return float(np.clip(base * noise, 0.3, 2.0))

def per_day_amount_jitter(discipline: float) -> float:
    """
    Gentle same-day ticket-size jitter; higher for low discipline.
    """
    sigma = 0.25 + 0.45 * (1.0 - float(np.clip(discipline, 0.0, 1.0)))
    return float(np.clip(np.random.lognormal(mean=0.0, sigma=sigma), 0.8, 1.6))


# ----- Quarter slicing -----
def quarter_windows(start: date, end: date):
    """Yield (q_start, q_end_exclusive) for each quarter overlapping [start, end]."""
    def q_start_of(d: date) -> date:
        return date(d.year, ((d.month-1)//3)*3 + 1, 1)
    d = start
    seen = set()
    while d <= end:
        qs = q_start_of(d)
        if qs not in seen:
            seen.add(qs)
            # next quarter start
            if qs.month in (1,4,7,10):
                if qs.month == 10:
                    nqs = date(qs.year+1, 1, 1)
                else:
                    nqs = date(qs.year, qs.month+3, 1)
            else:
                # should never happen, but be safe
                nqs = date(qs.year, ((qs.month-1)//3)*3 + 4, 1)
            yield (max(qs, start), min(nqs, end + timedelta(days=1)))
        # jump to next month start to find next quarter
        if d.month == 12:
            d = date(d.year+1, 1, 1)
        else:
            d = date(d.year, d.month+1, 1)

# ----- Quarter metrics & crash detection -----
@dataclass
class QuarterMetrics:
    min_balance: float = float('inf')
    neg_days: int = 0
    atm_total: float = 0.0
    spend_total: float = 0.0  # visible card purchases (optional)

def update_qmetrics(qm: QuarterMetrics, balance_after: float, ttype: int, amt: float):
    qm.min_balance = min(qm.min_balance, balance_after)
    if balance_after < 0:
        qm.neg_days += 1
    # 2 = ATM, 1 = purchase in your TT map
    if ttype == TT["atm_withdrawal"]:
        qm.atm_total += amt
    elif ttype == TT["purchase"]:
        qm.spend_total += amt

def crashed(qm: QuarterMetrics, monthly_salary: float) -> bool:
    """Crash if deep overdraft or many negative days in the quarter."""
    deep = qm.min_balance < (0.50 * monthly_salary)
    frequent = qm.neg_days >= 10  # ~>10 days in a ~90-day quarter------0.
    return bool(deep or frequent)

# ----- Strategy adjustment after crash -----
def adjust_discipline_for_next_quarter(disc: float, did_crash: bool) -> float:
    # If crashed → become more disciplined; else small random drift.
    if did_crash:
        disc = min(0.98, disc * 1.12 + 0.04)
    else:
        disc = float(np.clip(disc + np.random.normal(0.0, 0.02), 0.10, 0.98))
    return disc

def reweight_categories_for_quarter(cust_row: pd.Series, did_crash: bool) -> dict:
    """Return category weights (excluding 'atm') for this quarter."""
    cats, ws = [], []
    for suffix, cat in WEIGHT_COL_TO_CATEGORY.items():
        if cat == "atm":
            continue
        col = f"w_cat_{suffix}"
        if col in cust_row:
            cats.append(cat); ws.append(float(cust_row[col]))
    if not cats:
        return {"shopping": 1.0}
    w = np.array(ws, dtype=float)
    if did_crash:
        # Shift 20–30% away from discretionary to essentials
        idx_map = {c:i for i,c in enumerate(cats)}
        def pull(name, pct):
            if name in idx_map:
                w[idx_map[name]] *= (1.0 - pct)
        def push(name, pct):
            if name in idx_map:
                w[idx_map[name]] *= (1.0 + pct)
        pull("shopping", 0.25); pull("entertainment", 0.25); pull("restaurant", 0.15)
        push("grocery", 0.20); push("transportation", 0.15); push("gas", 0.10); push("healthcare", 0.10)
    else:
        # small natural drift each quarter to avoid stasis
        w *= np.random.uniform(0.95, 1.05, size=w.shape)
    w = np.maximum(w, 1e-6); w = w / w.sum()
    return {c: float(wi) for c, wi in zip(cats, w)}

def pick_category_with_override(cat_weights: dict) -> str:
    cats = list(cat_weights.keys()); w = np.array(list(cat_weights.values()), dtype=float)
    w = w / w.sum()
    return cats[np.random.choice(len(cats), p=w)]

def reweight_channels_for_quarter(cust_row: pd.Series, did_crash: bool) -> dict:
    """Return channel weights for this quarter ('pos','online','atm','branch')."""
    ch_names, ws = [], []
    for col, key in CHANNEL_COLUMNS.items():
        c = f"w_channel_{col}"
        if c in cust_row:
            ch_names.append(key); ws.append(float(cust_row[c]))
    if not ch_names:
        return {"pos":0.5, "online":0.5, "atm":0.0, "branch":0.0}
    w = np.array(ws, dtype=float)
    name_idx = {n:i for i,n in enumerate(ch_names)}
    if did_crash and "atm" in name_idx:
        # discourage heavy ATM if they crashed
        w[name_idx["atm"]] *= 0.1
        # nudge POS/online up
        if "pos" in name_idx: w[name_idx["pos"]] *= 1.10
        if "online" in name_idx: w[name_idx["online"]] *= 1.10
    else:
        w *= np.random.uniform(0.96, 1.04, size=w.shape)
    w = np.maximum(w, 1e-6); w = w / w.sum()
    return {n: float(wi) for n, wi in zip(ch_names, w)}

def pick_channel_with_override(ch_weights: dict, channels_df: pd.DataFrame) -> int:
    names = list(ch_weights.keys()); w = np.array(list(ch_weights.values()), dtype=float)
    w = w / w.sum()
    choice = names[np.random.choice(len(names), p=w)]
    m = channels_df.loc[channels_df["channel_name"].str.lower() == choice.lower()]
    return int(m["channel_id"].iloc[0]) if not m.empty else int(channels_df["channel_id"].iloc[0])

# ----- Caps builder limited to a window (quarter) -----
def build_caps_for_window(scheduled: dict,
                          window_start: date, window_end_excl: date,
                          monthly_salary: float, monthly_passive: float,
                          monthly_rent_in: float, monthly_rent_out: float,
                          discipline: float) -> dict:
    """Same logic as your build_monthly_caps, but only fill dates in [window_start, window_end_excl)."""
    from collections import defaultdict
    disc = float(np.clip(discipline, 0.10, 0.98))
    save_rate = savings_rate_from_discipline(disc)
    fixed_out = defaultdict(float); income_in = defaultdict(float)

    # Aggregate scheduled in window by month
    d_iter = window_start
    while d_iter < window_end_excl:
        if d_iter in scheduled:
            ym = (d_iter.year, d_iter.month)
            for e in scheduled[d_iter]:
                amt = float(e["amount"])
                if e["type_id"] in {TT["bill_payment"], TT["rent_out"]}: fixed_out[ym] += amt
                elif e["type_id"] in {TT["salary"], TT["passive_income"], TT["rent_in"]}: income_in[ym] += amt
        d_iter += timedelta(days=1)

    # Walk months that overlap window
    caps = {}
    cur = date(window_start.year, window_start.month, 1)
    last = date((window_end_excl - timedelta(days=1)).year, (window_end_excl - timedelta(days=1)).month, 1)
    while cur <= last:
        y, m = cur.year, cur.month
        inc = income_in.get((y, m), monthly_salary + monthly_passive + monthly_rent_in)
        fix = fixed_out.get((y, m), monthly_rent_out)
        savings = inc * save_rate
        discretionary = max(0.0, inc - fix - savings)
        # day weights within month (you already have month_daily_weights)
        if m == 12: next_first = date(y+1,1,1)
        else:       next_first = date(y, m+1, 1)
        ndays = (next_first - cur).days
        weights = month_daily_weights(cur, ndays, disc)
        for i in range(ndays):
            d = cur + timedelta(days=i)
            if d < window_start or d >= window_end_excl: continue
            caps[d] = discretionary * weights[d]
        cur = next_first
    return caps

def _compute_age(dob_val, on_date: date) -> int:
    try:
        dob = pd.to_datetime(dob_val).date() if not isinstance(dob_val, date) else dob_val
    except Exception:
        return 30
    return int(max(0, on_date.year - dob.year - ((on_date.month, on_date.day) < (dob.month, dob.day))))

def _edu_level(edu_val) -> int:
    s = str(edu_val or "").strip().lower()
    return EDU_LEVELS.get(s, 3)  # keep your EDU_LEVELS map as-is

def credit_factor_from_score(score) -> float:
    try:
        x = float(score)
        z = (x - 300.0) / (850.0 - 300.0)
        return float(np.clip(z, 0.0, 1.0))
    except Exception:
        return 0.5

def discipline_from_age_education_credit(dob_val, edu_val, credit_score, segment, ref_date: date) -> tuple[float, int, int, float]:
    """
    Higher variance version: non-linear age & credit shaping, segment multiplier, and a small idiosyncratic shock.
    Returns (discipline in [0.10, 0.98], age, edu_level, credit_norm)
    """
    age = _compute_age(dob_val, ref_date)
    lvl = _edu_level(edu_val)
    edu_base = EDU_BASELINE_DISCIPLINE.get(lvl, 0.30)

    # Non-linear age factor: 18->0 ... 72->1, then sharpen differences
    age_lin = np.clip((age - 18) / (72 - 18), 0.0, 1.0)
    age_factor = float(age_lin ** 1.35)  # exponent > 1 widens spread

    # Non-linear credit factor: amplify differences away from middle
    credit_norm = credit_factor_from_score(credit_score)
    credit_factor = float(credit_norm ** 1.25)

    # Blend weights (more balanced, no big constant offset)
    w_edu, w_age, w_credit = 0.30, 0.40, 0.30
    disc_raw = (w_edu * edu_base) + (w_age * age_factor) + (w_credit * credit_factor)

    # Segment multiplier (cohort tilt)
    seg_key = str(segment or "").strip().lower()
    seg_mult = SEG_DISC_MULTIPLIER.get(seg_key, 1.0)
    disc_raw *= seg_mult

    # Idiosyncratic shock (stable-ish): variance higher for lower edu/credit
    # σ ranges ~0.02..0.14
    sigma = 0.02 + 0.07 * (1.0 - edu_base) + 0.04 * (1.0 - credit_factor)
    eps = float(np.random.normal(0.0, sigma))

    # Contrast stretch around 0.5 to widen separation, then clamp
    k = 1.35  # increase to widen further (1.1..1.6 reasonable)
    disc_stretched = 0.5 + k * ((disc_raw + eps) - 0.5)

    disc = float(np.clip(disc_stretched, 0.10, 0.98))
    return disc, age, lvl, credit_norm


def savings_rate_from_discipline(disc: float) -> float:
    """
    Looser mapping so outcomes spread more:
    - Low discipline: ~2–5% savings
    - High discipline: ~18–24% savings
    """
    d = float(np.clip(disc, 0.0, 1.0))
    # sigmoid spreads the middle so low/high differ more
    s = 1.0 / (1.0 + np.exp(-4.0 * (d - 0.5)))   # ~0.018 .. ~0.982
    return 0.02 + 0.22 * s




def month_daily_weights(start_d: date, ndays: int, disc: float) -> Dict[date, float]:
    """
    Smooth per-day weights across the month.
    Slight weekend bump only if discipline is low.
    """
    weights = np.ones(ndays, dtype=float)
    weekend_bump = 1.0 + 0.1 * (1.0 - disc)    # up to +6% on weekends for low-discipline
    for i in range(ndays):
        d = start_d + timedelta(days=i)
        if d.weekday() >= 5:  # 5=Sat,6=Sun
            weights[i] *= weekend_bump
    weights = weights / weights.sum()
    return {start_d + timedelta(days=i): float(weights[i]) for i in range(ndays)}

def expected_ticket_amount(cust: pd.Series, seg_ranges: Dict[str, Tuple[float,float]], monthly_salary: float) -> float:
    """
    Estimate average card-purchase ticket using per-customer category weights and segment ranges.
    """
    tots = 0.0; wsum = 0.0
    for suffix, cat in WEIGHT_COL_TO_CATEGORY.items():
        if cat == "atm":
            continue  # exclude ATM from 'purchase' ticket
        col = f"w_cat_{suffix}"
        if col in cust:
            w = float(cust[col])
            lo, hi = seg_ranges.get(cat, DEFAULT_CATEGORY_RANGES.get(cat, (10, 120)))
            med = 0.5 * (lo + hi)
            tots += w * med
            wsum += w
    if wsum <= 0:
        lo, hi = DEFAULT_CATEGORY_RANGES["shopping"]
        tots = 0.5 * (lo + hi); wsum = 1.0
    inc_scale = 1.0 + min(0.8, monthly_salary / 8000.0)
    return max(8.0, (tots / wsum) * inc_scale)

def build_monthly_caps(scheduled: dict,
                       start: date, end: date,
                       monthly_salary: float, monthly_passive: float,
                       monthly_rent_in: float, monthly_rent_out: float,
                       discipline: float) -> dict[date, float]:
    """
    For each month in range, compute per-day discretionary card-spend caps
    using fixed items (salary/passive/rent/bills) and savings derived from discipline.
    """
    disc = float(np.clip(discipline, 0.20, 0.90))
    save_rate = savings_rate_from_discipline(disc)

    from collections import defaultdict
    fixed_out = defaultdict(float)
    income_in = defaultdict(float)

    for d, evts in scheduled.items():
        ym = (d.year, d.month)
        for e in evts:
            amt = float(e["amount"])
            if e["type_id"] in {TT["bill_payment"], TT["rent_out"]}:
                fixed_out[ym] += amt
            elif e["type_id"] in {TT["salary"], TT["passive_income"], TT["rent_in"]}:
                income_in[ym] += amt

    # months in range
    months = []
    cur = date(start.year, start.month, 1)
    last = date(end.year, end.month, 1)
    while cur <= last:
        months.append(cur)
        y, m = cur.year, cur.month
        cur = date(y + (m // 12), (m % 12) + 1, 1)

    caps = {}
    for first_day in months:
        y, m = first_day.year, first_day.month
        inc = income_in.get((y, m), monthly_salary + monthly_passive + monthly_rent_in)
        fix = fixed_out.get((y, m), monthly_rent_out)
        savings = inc * save_rate
        discretionary = max(0.0, inc - fix - savings)

        # per-day weights within the month depend on discipline (weekend bump if low)
        if m == 12:
            next_first = date(y + 1, 1, 1)
        else:
            next_first = date(y, m + 1, 1)
        ndays = (next_first - first_day).days
        weights = month_daily_weights(first_day, ndays, disc)  # existing helper

        for i in range(ndays):
            d = first_day + timedelta(days=i)
            if d < start or d > end:
                continue
            caps[d] = discretionary * weights[d]
    return caps



def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def to_bigint_id(any_id) -> int:
    s = str(any_id)
    digits = "".join(ch for ch in s if ch.isdigit())
    return int(digits) if digits else abs(hash(s)) % 10**11

def yyyymmdd_int(d: date) -> int:
    return d.year * 10000 + d.month * 100 + d.day

def clamp_day(year: int, month: int, day: int) -> date:
    if month == 12:
        last = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        last = date(year, month + 1, 1) - timedelta(days=1)
    return date(year, month, min(day, last.day))

def parse_bool(v) -> bool:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return False
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    return str(v).strip().lower() in {"true", "t", "1", "yes", "y"}

def safe_day_from_cell(v, default_day: int) -> int:
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)) or (isinstance(v, str) and v.strip() == ""):
            return int(default_day)
        day = int(float(v))
        return max(1, min(28, day))
    except Exception:
        return int(default_day)

def due_day_if_flag(flag_val, day_cell, default_day: int):
    flag = parse_bool(flag_val)
    if not flag:
        return None
    return safe_day_from_cell(day_cell, default_day)

def sample_time_from_normal(mean_hour: float, std_hour: float, min_hour: float = 6.0) -> time:
    h = np.random.normal(loc=float(mean_hour), scale=max(0.1, float(std_hour)))
    h = max(min_hour, min(23.0, h))
    hour = int(h)
    minutes = int((h - hour) * 60)
    seconds = int((h - hour - minutes / 60) * 3600)
    return time(hour, minutes, seconds)

def weighted_choice(items: List, weights: List[float]):
    w = np.array(weights, dtype=float)
    w = w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(items)
    return items[np.random.choice(len(items), p=w)]

def next_txn_id(seq_num: int) -> str:
    return f"txn{seq_num:011d}"

def get_ranges_for_segment(segment: str) -> Dict[str, Tuple[float, float]]:
    return SEGMENT_CATEGORY_RANGES.get(str(segment).strip().lower(), DEFAULT_CATEGORY_RANGES)

def amount_from_range(rng: Tuple[float, float]) -> float:
    lo, hi = rng
    u = np.random.beta(2, 5)  # skew small, occasional big
    return lo + (hi - lo) * u

def income_scale(monthly_salary: float) -> float:
    return 1.0 + min(0.3, monthly_salary / 8000.0)

def daily_thin_factor(curr_balance: float, monthly_salary: float, cfg: DEFAULTS) -> float:
    """thr_hi = monthly_salary * cfg.thin_threshold_hi
    thr_lo = monthly_salary * cfg.thin_threshold_lo
    if curr_balance < thr_lo:
        return cfg.thin_factor_lo
    if curr_balance < thr_hi:
        return cfg.thin_factor_hi"""
    return 1.0

def amount_spend_pressure(curr_balance: float, monthly_salary: float, cfg: DEFAULTS) -> float:
    thr_hi = monthly_salary * cfg.thin_threshold_hi
    thr_lo = monthly_salary * cfg.thin_threshold_lo
    if curr_balance < thr_lo:
        return cfg.amount_pressure_lo
    if curr_balance < thr_hi:
        return cfg.amount_pressure_hi
    return 1.0

def choose_random_merchant(merchants_df: pd.DataFrame, categories_df: pd.DataFrame, category_name: str) -> Optional[int]:
    cand = categories_df.loc[categories_df["category_name"].str.lower() == category_name.lower(), "category_id"]
    if cand.empty:
        return int(merchants_df["merchant_id"].sample(1).iloc[0]) if len(merchants_df) else None
    mcat = int(cand.iloc[0])
    m = merchants_df.loc[merchants_df["category_id"] == mcat]
    if m.empty:
        return int(merchants_df["merchant_id"].sample(1).iloc[0]) if len(merchants_df) else None
    return int(m["merchant_id"].sample(1).iloc[0])

def pick_channel_id(row: pd.Series, channels_df: pd.DataFrame) -> int:
    names, weights = [], []
    for col, key in CHANNEL_COLUMNS.items():
        colname = f"w_channel_{col}"
        if colname in row:
            names.append(key)
            weights.append(float(row[colname]))
    choice = weighted_choice(names, weights)
    match = channels_df.loc[channels_df["channel_name"].str.lower() == choice.lower()]
    return int(match["channel_id"].iloc[0]) if not match.empty else int(channels_df["channel_id"].iloc[0])

def channel_name_by_id(channels_df: pd.DataFrame, channel_id: int) -> str:
    m = channels_df.loc[channels_df["channel_id"] == channel_id]
    return str(m["channel_name"].iloc[0]) if not m.empty else str(channels_df["channel_name"].iloc[0])

# ---------- PRICE DISPERSION, MIXTURES, SEASONALITY ----------

def build_merchant_price_index(merchants_df: pd.DataFrame) -> dict[int, float]:
    """
    Assign a persistent 'price index' to each merchant: ~ LogNormal(0, 0.2), clamped [0.75, 1.6].
    Higher index → pricier baskets at that merchant.
    """
    out = {}
    for _, r in merchants_df.iterrows():
        mid = int(r["merchant_id"])
        idx = float(np.clip(np.random.lognormal(mean=0.0, sigma=0.20), 0.75, 1.60))
        out[mid] = idx
    return out

def _cat_tier_mix(seg_ranges: dict[str, tuple[float,float]], cat: str):
    """
    For a given category, produce 3 tiers within that segment's [lo,hi]:
      - small (≈ everyday), medium, big (rare)
    """
    lo, hi = seg_ranges.get(cat, DEFAULT_CATEGORY_RANGES.get(cat, (10, 120)))
    span = max(1.0, hi - lo)
    tiers = [
        (0.58, (lo,               lo + 0.45 * span)),  # small
        (0.32, (lo + 0.25 * span, lo + 0.80 * span)),  # medium
        (0.10, (lo + 0.70 * span, hi)),                # big
    ]
    # Normalize probabilities (guarding numerical drift)
    ps = np.array([p for p, _ in tiers], dtype=float)
    ps = ps / ps.sum()
    return list(zip(ps, [r for _, r in tiers]))

def _triangular(a: float, b: float) -> float:
    """Sample with mode closer to lower end (more small baskets than big)."""
    mode = a + 0.30 * (b - a)
    if a < b:
        return float(np.random.triangular(a, mode, b))
    else:
        temp = a
        a = b
        b = temp
        return float(np.random.triangular(a, mode, b))

def _dom_factor(day: int) -> float:
    """Day-of-month factor: slightly larger early month, slightly smaller at month-end."""
    if day <= 5:
        return 1.08
    if day >= 26:
        return 0.97
    return 1.00

def _dow_factor(dow: int) -> float:
    """Weekend bump for discretionary categories."""
    return 1.04 if dow >= 5 else 1.00

def _tx_variance_from_discipline(discipline: float) -> float:
    """
    LogNormal sigma: higher for low-discipline -> more ticket volatility.
    Range ≈ [0.25 .. 0.70]
    """
    return 0.25 + 0.45 * (1.0 - float(np.clip(discipline, 0.0, 1.0)))

def _merchant_jitter(idx: float) -> float:
    """Small per-transaction jitter around the merchant's price index."""
    return float(np.clip(np.random.normal(loc=idx, scale=0.05 * idx), 0.60, 1.90))

def _maybe_big_ticket_boost(cat: str, discipline: float, base_hi: float) -> float:
    """
    Rare big-ticket shock (electronics, medical, etc.).
    More likely when discipline is low; capped by category upper band.
    """
    base_p = 0.01
    if cat in {"shopping", "healthcare", "entertainment"}:
        base_p = 0.02
    p = base_p + 0.05 * (1.0 - discipline)  # up to ~7% for low-discipline
    if np.random.rand() < p:
        return float(np.random.uniform(1.05, 1.35))  # 5%..35% extra
    return 1.0

def sample_purchase_amount(
    category: str,
    seg_ranges: dict[str, tuple[float, float]],
    discipline: float,
    merchant_idx: float,
    day_obj: date,
) -> float:
    """
    Draw a purchase ticket with rich variability:
      tiered category ranges × lognormal variance × merchant index × seasonality × occasional big-ticket.
    """
    # 1) choose tier
    tiers = _cat_tier_mix(seg_ranges, category)
    which = np.random.choice(len(tiers), p=np.array([p for p, _ in tiers]))
    a, b = tiers[which][1]

    # 2) base within tier (triangular)
    base = _triangular(a, b)

    # 3) person variance (lognormal) — discipline reduces sigma
    sigma = _tx_variance_from_discipline(discipline)
    var_mult = float(np.random.lognormal(mean=0.0, sigma=sigma))

    # 4) day of month & day of week
    dom_mult = _dom_factor(day_obj.day)
    dow_mult = _dow_factor(day_obj.weekday())

    # 5) merchant effect (+ tiny jitter)
    merch_mult = _merchant_jitter(merchant_idx)

    # 6) occasional big-ticket boost
    shock_mult = _maybe_big_ticket_boost(category, discipline, b)

    amt = base * var_mult * dom_mult * dow_mult * merch_mult * shock_mult
    return round(max(1.0, amt), 2)

def soft_cap(amount: float, remaining: float, slack: float, discipline: float) -> tuple[bool, float]:
    """
    Soften the day cap:
      - If amount within (1+slack)*remaining → allow (possibly shrink a bit)
      - If far above → either skip (likely) or shrink near remaining (rarely), depending on discipline
    Returns (keep, maybe_adjusted_amount)
    """
    if remaining <= 2.0:
        return False, amount

    if amount <= remaining:
        # small random nudges to avoid exact repeats
        return True, round(amount * np.random.uniform(0.98, 1.02), 2)

    # over remaining
    overshoot = amount / remaining
    if overshoot <= (1.0 + slack):
        # accept but shrink to something near remaining (avoid exact remaining)
        return True, round(remaining * np.random.uniform(0.85, 0.995), 2)

    # way over remaining → usually skip; disciplined customers more likely to skip
    skip_prob = 0.65 + 0.25 * discipline  # 65%..90%
    if np.random.rand() < skip_prob:
        return False, amount
    # rare keep: shrink heavily near remaining
    return True, round(remaining * np.random.uniform(0.75, 0.95), 2)



## ----- Main generator -----

def generate_transactions(customers_df: pd.DataFrame,
                          merchants_df: pd.DataFrame,
                          channels_df: pd.DataFrame,
                          categories_df: pd.DataFrame,
                          branch_id: int,
                          start_date: str = DEFAULTS.start_date,
                          end_date: str = DEFAULTS.end_date,
                          currency: str = DEFAULTS.currency,
                          random_state: int = DEFAULTS.random_state) -> pd.DataFrame:
    """
    Quarterly-adaptive generator (with ATM & Bills merchants, is_salary column, and integer merchant IDs).
    - Simulates within [start_date, end_date]
    - Recomputes discipline & strategy each quarter, reacts to 'crashes'
    - Budget-aware (daily caps), cash-shadow for ATM, spend pressure at low balance
    - Segment-specific price ranges; varied purchase tickets
    - Uses category merchants for 'atm' and 'Bills and Payment'
    - Rent in/out & salary/passive use merchant_id = 0 (integer)
    - Adds 'is_salary' flag (True only for salary)
    """
    # --------- setup & safe fallbacks ---------
    cfg = DEFAULTS()
    cfg.start_date, cfg.end_date, cfg.currency, cfg.random_state = start_date, end_date, currency, random_state
    set_seed(cfg.random_state)

    start = pd.to_datetime(cfg.start_date).date()
    end   = pd.to_datetime(cfg.end_date).date()

    categories_df = categories_df.copy()
    if "category_name" in categories_df.columns:
        categories_df["category_name"] = categories_df["category_name"].astype(str)

    # merchant price index (fallback to 1.0 if helper missing)
    try:
        merchant_price_index = build_merchant_price_index(merchants_df)
    except Exception:
        merchant_price_index = {}
        if "merchant_id" in merchants_df.columns:
            for mid in merchants_df["merchant_id"]:
                merchant_price_index[int(mid)] = 1.0

    # global price lift (fallback to 1.0 if not defined)
    price_level = globals().get("PRICE_LEVEL", 1.0)

    # Helper: robust merchant picker by category name (case-insensitive)
    def pick_merchant_id_for_category(cat_name: str) -> int:
        try:
            cand_cat = categories_df.loc[
                categories_df["category_name"].str.lower() == str(cat_name).strip().lower(), "category_id"
            ]
            if not cand_cat.empty:
                cat_id = int(cand_cat.iloc[0])
                m = merchants_df.loc[merchants_df["category_id"] == cat_id, "merchant_id"]
                if not m.empty:
                    return int(m.sample(1).iloc[0])
        except Exception:
            pass
        return 0  # safe integer fallback

    rows: list[dict] = []
    seq = 1

    # --------- per customer ---------
    for _, cust in customers_df.iterrows():
        # ids
        cust_id_big = to_bigint_id(cust["customer_id"])
        account_id_big = cust_id_big

        # segment & ranges
        seg = str(cust.get("segment", "")).strip().lower()
        try:
            seg_ranges = get_ranges_for_segment(seg)
        except Exception:
            seg_ranges = DEFAULT_CATEGORY_RANGES

        # income & fixeds
        monthly_salary   = float(cust.get("monthly_salary_usd", 0.0))
        monthly_passive  = float(cust.get("monthly_passive_income_usd", 0.0))
        monthly_rent_in  = float(cust.get("monthly_rent_in_usd", 0.0))
        monthly_rent_out = float(cust.get("monthly_rent_out_usd", 0.0))
        annual_income    = float(cust.get("annual_income", 12 * monthly_salary))

        # flags & due days
        bill_resp   = parse_bool(cust.get("bill_responsible", False))
        is_renter   = parse_bool(cust.get("is_renter", False))
        bills_due_day = due_day_if_flag(bill_resp, cust.get("bills_due_day"), default_day=10)
        rent_due_day  = due_day_if_flag(is_renter, cust.get("rent_due_day"),  default_day=5)
        salary_day    = safe_day_from_cell(cust.get("monthly_salary_day", 1), default_day=1)

        # behavior params
        weekly_rate = float(cust.get("weekly_txn_rate", 15.0))
        print(weekly_rate)
        tod_mean    = float(cust.get("tod_mean_hour", 13.0))
        tod_std     = float(cust.get("tod_std_hour", 3.0))

        # balances & state
        current_balance = float(annual_income) / 15.0
        cash_shadow = 0.0

        # ------------- 1) schedule recurring events for whole period -------------
        scheduled: dict[date, list[dict]] = {}
        month_cursor = date(start.year, start.month, 1)
        while month_cursor <= end:
            y, m = month_cursor.year, month_cursor.month

            # salary (merchant_id=0, is_salary=True)
            s_day = clamp_day(y, m, salary_day)
            if start <= s_day <= end and monthly_salary > 0:
                _add_sched(scheduled, s_day, {
                    "time": cfg.salary_time, "type_id": TT["salary"],
                    "amount": round(monthly_salary, 2),
                    "merchant_id": 0, "channel": "branch",
                    "is_recurring": True, "is_salary": True
                })

            # passive income (merchant_id=0, is_salary=False)
            if monthly_passive > 0:
                p_day = clamp_day(y, m, 20)
                if start <= p_day <= end:
                    _add_sched(scheduled, p_day, {
                        "time": cfg.passive_time, "type_id": TT["passive_income"],
                        "amount": round(monthly_passive, 2),
                        "merchant_id": 0, "channel": "online",
                        "is_recurring": True, "is_salary": False
                    })

            # rent in (merchant_id=0)
            if monthly_rent_in > 0:
                rin_day = clamp_day(y, m, 5)
                if start <= rin_day <= end:
                    _add_sched(scheduled, rin_day, {
                        "time": cfg.rent_in_time, "type_id": TT["rent_in"],
                        "amount": round(monthly_rent_in, 2),
                        "merchant_id": 0, "channel": "online",
                        "is_recurring": True, "is_salary": False
                    })

            # rent out (merchant_id=0)
            if rent_due_day is not None and monthly_rent_out > 0:
                rout_day = clamp_day(y, m, rent_due_day)
                if start <= rout_day <= end:
                    _add_sched(scheduled, rout_day, {
                        "time": cfg.rent_time, "type_id": TT["rent_out"],
                        "amount": round(monthly_rent_out, 2),
                        "merchant_id": 0, "channel": "online",
                        "is_recurring": True, "is_salary": False
                    })

            # bills (use your "Bills and Payment" merchants)
            if bills_due_day is not None and monthly_salary > 0:
                b_day = clamp_day(y, m, bills_due_day)
                if start <= b_day <= end:
                    for _, amt in _bill_amounts(monthly_salary):
                        bill_merchant_id = pick_merchant_id_for_category("Bill and Payments")
                        _add_sched(scheduled, b_day, {
                            "time": cfg.bills_time, "type_id": TT["bill_payment"],
                            "amount": round(amt, 2),
                            "merchant_id": int(bill_merchant_id), "channel": "online",
                            "is_recurring": True, "is_salary": False
                        })

            # next month
            month_cursor = date(y + (m // 12), (m % 12) + 1, 1)

        # average ticket (for cap→λ limiting)
        try:
            avg_ticket = expected_ticket_amount(cust, seg_ranges, monthly_salary)
        except Exception:
            lo, hi = seg_ranges.get("shopping", (15, 300))
            avg_ticket = max(8.0, 0.5 * (lo + hi) * income_scale(monthly_salary))

        # baseline discipline from age + education + credit + segment
        try:
            disc_base, age_years, edu_level, credit_norm = discipline_from_age_education_credit(
                cust.get("date_of_birth"),
                cust.get("education_level"),
                cust.get("credit_score"),
                cust.get("segment"),
                end  # reference date
            )
        except Exception:
            disc_base = 0.65

        # ------------- 2) quarterly simulation -------------
        q_prev_crashed = False
        disc_q = disc_base
        weekly_rate_q = weekly_rate

        for qs, qe_excl in quarter_windows(start, end):
            # adapt discipline & intensity based on last quarter result
            if q_prev_crashed:
                disc_q = adjust_discipline_for_next_quarter(disc_q, did_crash=True)
                weekly_rate_q *= 0.90
            else:
                disc_q = adjust_discipline_for_next_quarter(disc_q, did_crash=False)
                weekly_rate_q *= np.random.uniform(0.98, 1.02)

            # quarter-specific daily caps
            daily_caps_q = build_caps_for_window(
                scheduled=scheduled,
                window_start=qs, window_end_excl=qe_excl,
                monthly_salary=monthly_salary, monthly_passive=monthly_passive,
                monthly_rent_in=monthly_rent_in, monthly_rent_out=monthly_rent_out,
                discipline=disc_q
            )

            # quarterly category/channel reweights
            try:
                cat_w_q = reweight_categories_for_quarter(cust, did_crash=q_prev_crashed)
            except Exception:
                cat_w_q = {}
                for suffix, cat in WEIGHT_COL_TO_CATEGORY.items():
                    if cat == "ATM":
                        continue
                    col = f"w_cat_{suffix}"
                    if col in cust:
                        cat_w_q[cat] = float(cust[col])
                if not cat_w_q:
                    cat_w_q = {"shopping": 1.0}

            try:
                ch_w_q = reweight_channels_for_quarter(cust, did_crash=q_prev_crashed)
            except Exception:
                print("exception on quarter weights")
                ch_w_q = {}
                for col, key in CHANNEL_COLUMNS.items():
                    c = f"w_channel_{col}"
                    if c in cust:
                        ch_w_q[key] = float(cust[c])
                if not ch_w_q:
                    ch_w_q = {"pos": 0.4, "online": 0.4, "atm": 0.1, "branch": 0.1}
                else:
                    w = np.array(list(ch_w_q.values()), dtype=float)
                    if w.sum() > 0:
                        w = w / w.sum()
                        ch_w_q = dict(zip(ch_w_q.keys(), map(float, w)))

            # quarter-specific base daily rate (±10%)
            mu_week = np.random.normal(loc=weekly_rate_q, scale=max(0.01, weekly_rate_q * 0.10))
            base_daily_rate_q = max(0.0, mu_week) / 7.0

            # metrics for crash detection
            qm = QuarterMetrics()

            # iterate days in [qs, qe_excl)
            d = qs
            while d < qe_excl:
                # decay cash shadow each day
                cash_shadow *= cfg.cash_shadow_decay
                cash_shadow = min(cfg.cash_shadow_cap, max(0.0, cash_shadow))

                # 2.1 post recurring (salary first by time)
                if d in scheduled:
                    for evt in sorted(scheduled[d], key=lambda e: e["time"]):
                        prev = round(current_balance, 2)
                        amt = float(evt["amount"])
                        if evt["type_id"] in CREDIT_TYPES:
                            current_balance += amt
                        else:
                            current_balance -= amt
                        newb = round(current_balance, 2)

                        rows.append({
                            "transaction_id_dd": next_txn_id(seq),
                            "transaction_date": d.isoformat(),
                            "transaction_time": evt["time"].strftime("%H:%M:%S"),
                            "transaction_date_key": yyyymmdd_int(d),
                            "customer_id": cust_id_big,
                            "account_id": account_id_big,
                            "merchant_id": int(evt["merchant_id"]) if evt["merchant_id"] is not None else 0,
                            "branch_id": int(branch_id),
                            "channel_id": _channel_id_for_name(channels_df, evt["channel"]),
                            "transaction_type_id": int(evt["type_id"]),
                            "amount": round(amt, 2),
                            "currency": cfg.currency,
                            "is_recurring": True,
                            "is_salary": bool(evt.get("is_salary", False)),
                            "previous_balance": prev,
                            "balance_after_transaction": newb,
                        })
                        seq += 1
                        update_qmetrics(qm, newb, evt["type_id"], amt)

                # 2.2 variable spending (budget-aware)
                # ---- variable spending driven by weekly_txn_rate ----
                cap_today = float(daily_caps_q.get(d, 0.0))
                cap_purchase_today = cap_today * (1.0 - cfg.cash_shadow_amount_weight * cash_shadow)

                # Split weekly target into purchase vs ATM using the quarter's channel weights
                weekly_atm_target, weekly_purch_target = split_weekly_rate(weekly_rate_q, ch_w_q)
                lam_purch_target = max(0.0, weekly_purch_target / 7.0)
                lam_atm_target   = max(0.0, weekly_atm_target   / 7.0)

                # Light-touch multipliers (discipline/balance/cash only *nudge* counts)
                m_p = purchase_count_multiplier(current_balance, monthly_salary, d, cash_shadow, disc_q, cfg)
                m_a = atm_count_multiplier(current_balance, monthly_salary, d, cash_shadow, disc_q, q_prev_crashed)

                lam_p = lam_purch_target * m_p
                lam_a = lam_atm_target   * m_a

                # Cap can still limit visible purchases (but cannot limit ATM frequency)
                expected_count_cap = cap_purchase_today / max(1.0, avg_ticket)
                lam_p = min(lam_p, max(0.0, expected_count_cap * 1.25))  # slightly looser than before

                k_p = int(np.random.poisson(lam=lam_p)) if lam_p > 0 else 0
                k_a = int(np.random.poisson(lam=lam_a)) if lam_a > 0 else 0

                atm_withdraw_total = 0.0
                var_spend_today = 0.0
                slack = 0.15
                amt_jit = per_day_amount_jitter(disc_q)

                # -- PURCHASES first (respecting cap) --
                for _ in range(k_p):
                    channel_id = pick_channel_with_override(ch_w_q, channels_df)
                    # if it randomly picked ATM, force a purchase channel instead
                    if channel_name_by_id(channels_df, channel_id).lower() == "atm":
                        # choose POS or online by relative weight
                        alt = {k:v for k,v in ch_w_q.items() if k in ("pos","online","branch")}
                        if not alt: alt = {"pos":1.0}
                        channel_id = pick_channel_with_override(alt, channels_df)

                    t = sample_time_from_normal(tod_mean, tod_std, min_hour=10.0)
                    ttype = TT["purchase"]

                    # category by quarter weights + merchant
                    try:
                        category = pick_category_with_override(cat_w_q)
                    except Exception:
                        cats, ws = [], []
                        for suffix, cat in WEIGHT_COL_TO_CATEGORY.items():
                            if cat == "atm": continue
                            col = f"w_cat_{suffix}"
                            if col in cust: cats.append(cat); ws.append(float(cust[col]))
                        category = weighted_choice(cats, ws) if cats else "shopping"

                    merchant_id = choose_random_merchant(merchants_df, categories_df, category)
                    merchant_id = int(merchant_id) if merchant_id is not None else 0
                    m_idx = merchant_price_index.get(int(merchant_id), 1.0)

                    # amount sampler → spend pressure → jitter → price level
                    try:
                        amt = sample_purchase_amount(category, seg_ranges, disc_q, m_idx, d)
                    except Exception:
                        lo, hi = seg_ranges.get(category, DEFAULT_CATEGORY_RANGES.get(category, (10, 120)))
                        base_amt = amount_from_range((lo, hi)) * income_scale(monthly_salary)
                        amt = max(1.0, base_amt * m_idx)

                    press = amount_spend_pressure(current_balance, monthly_salary, cfg)
                    amt = max(1.0, amt * press * amt_jit * price_level)

                    # soft cap for purchases
                    if cap_purchase_today > 0:
                        remaining = cap_purchase_today * (1.0 + slack) - var_spend_today
                        keep, adj_amt = soft_cap(amt, remaining, slack=slack, discipline=disc_q)
                        if not keep:
                            continue
                        amt = adj_amt
                    var_spend_today += amt
                    amt = round(amt, 2)

                    prev = round(current_balance, 2)
                    current_balance -= amt
                    newb = round(current_balance, 2)

                    rows.append({
                        "transaction_id_dd": next_txn_id(seq),
                        "transaction_date": d.isoformat(),
                        "transaction_time": t.strftime("%H:%M:%S"),
                        "transaction_date_key": yyyymmdd_int(d),
                        "customer_id": cust_id_big,
                        "account_id": account_id_big,
                        "merchant_id": int(merchant_id),
                        "branch_id": int(branch_id),
                        "channel_id": int(channel_id),
                        "transaction_type_id": int(ttype),
                        "amount": round(amt, 2),
                        "currency": cfg.currency,
                        "is_recurring": False,
                        "is_salary": False,
                        "previous_balance": prev,
                        "balance_after_transaction": newb,
                    })
                    seq += 1
                    update_qmetrics(qm, newb, ttype, amt)

                # -- ATMs next (unaffected by purchase cap) --
                for _ in range(k_a):
                    channel_id = pick_channel_with_override(ch_w_q, channels_df)
                    # force ATM channel for ATM lambda
                    if channel_name_by_id(channels_df, channel_id).lower() != "atm":
                        channel_id = _channel_id_for_name(channels_df, "atm")

                    t = sample_time_from_normal(tod_mean, tod_std, min_hour=10.0)
                    ttype = TT["atm_withdrawal"]
                    rng = seg_ranges.get("atm", DEFAULT_CATEGORY_RANGES["atm"])
                    amt = round(max(5.0, amount_from_range(rng) * income_scale(monthly_salary) * price_level), 2)
                    merchant_id = pick_merchant_id_for_category("atm")
                    merchant_id = int(merchant_id) if isinstance(merchant_id, (int, np.integer)) else 0

                    prev = round(current_balance, 2)
                    current_balance -= amt
                    newb = round(current_balance, 2)

                    rows.append({
                        "transaction_id_dd": next_txn_id(seq),
                        "transaction_date": d.isoformat(),
                        "transaction_time": t.strftime("%H:%M:%S"),
                        "transaction_date_key": yyyymmdd_int(d),
                        "customer_id": cust_id_big,
                        "account_id": account_id_big,
                        "merchant_id": int(merchant_id),
                        "branch_id": int(branch_id),
                        "channel_id": int(channel_id),
                        "transaction_type_id": int(ttype),
                        "amount": round(amt, 2),
                        "currency": cfg.currency,
                        "is_recurring": False,
                        "is_salary": False,
                        "previous_balance": prev,
                        "balance_after_transaction": newb,
                    })
                    seq += 1
                    update_qmetrics(qm, newb, ttype, amt)
                    atm_withdraw_total += amt


                # end-of-day: cash shadow grows with today's ATM
                if atm_withdraw_total > 0 and monthly_salary > 0:
                    gain = atm_withdraw_total / (monthly_salary * cfg.cash_shadow_gain_div_salary)
                    gain = max(0.10, min(cfg.cash_shadow_cap, gain))
                    cash_shadow = min(cfg.cash_shadow_cap, cash_shadow + gain)

                # advance one day
                d += timedelta(days=1)

            # decide for next quarter
            q_prev_crashed = crashed(qm, monthly_salary)

        # end per-customer loop

    # --------- finalize dataframe ---------
    out_cols = [
        "transaction_id_dd",
        "transaction_date",
        "transaction_time",
        "transaction_date_key",
        "customer_id",
        "account_id",
        "merchant_id",
        "branch_id",
        "channel_id",
        "transaction_type_id",
        "amount",
        "currency",
        "is_recurring",
        "is_salary",
        "previous_balance",
        "balance_after_transaction",
    ]
    df = pd.DataFrame(rows, columns=out_cols)

    # ensure integer dtype for merchant_id where possible
    try:
        df["merchant_id"] = df["merchant_id"].fillna(0).astype(int)
    except Exception:
        pass

    return df






# =========================
# Utilities used above
# =========================

def _channel_id_for_name(channels_df: pd.DataFrame, name_lower: str) -> int:
    m = channels_df.loc[channels_df["channel_name"].str.lower() == name_lower.lower()]
    return int(m["channel_id"].iloc[0]) if not m.empty else int(channels_df["channel_id"].iloc[0])

def _add_sched(scheduled: Dict[date, List[dict]], d: date, evt: dict):
    scheduled.setdefault(d, []).append(evt)

def _bill_amounts(monthly_salary: float) -> List[Tuple[str, float]]:
    comps = [("Electricity", 0.03), ("Water", 0.01), ("Internet", 0.015), ("Mobile", 0.012)]
    k = np.random.randint(2, 5)  # 2..4 bills
    choices = random.sample(comps, k)
    out = []
    for _, pct in choices:
        out.append(("bill", max(5.0, monthly_salary * pct * np.random.uniform(0.9, 1.1))))
    return out


# ---------------------------
# Example usage
# ---------------------------

if __name__ == "__main__":
    # Example: assume you already loaded these four DataFrames
    customers_df = pd.read_csv("customers.csv")
    merchants_df = pd.read_csv("merchants.csv")
    channels_df  = pd.read_csv("channels.csv")
    categories_df = pd.read_csv("categories.csv")

    df = generate_transactions(customers_df, merchants_df, channels_df, categories_df,
                               branch_id=501,
                               start_date="2024-01-01", end_date="2024-12-28")
    df.to_csv("transactions_2024.csv", index=False)
    pass
