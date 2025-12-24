def generate_transactions(customers_df: pd.DataFrame,
                          merchants_df: pd.DataFrame,
                          channels_df: pd.DataFrame,
                          categories_df: pd.DataFrame,
                          branch_id: int,
                          start_date: str = DEFAULTS.start_date,
                          end_date: str = DEFAULTS.end_date,
                          currency: str = DEFAULTS.currency,
                          random_state: int = DEFAULTS.random_state) -> pd.DataFrame:

    cfg = DEFAULTS()
    cfg.start_date, cfg.end_date, cfg.currency, cfg.random_state = start_date, end_date, currency, random_state
    set_seed(cfg.random_state)

    start = pd.to_datetime(cfg.start_date).date()
    end   = pd.to_datetime(cfg.end_date).date()
    all_days = pd.date_range(start, end, freq="D").date

    rows=[]; seq=1
    categories_df = categories_df.copy()
    categories_df["category_name"] = categories_df["category_name"].astype(str)
    merchant_price_index = build_merchant_price_index(merchants_df)


    for _, cust in customers_df.iterrows():
        cust_id_big = to_bigint_id(cust["customer_id"])
        account_id_big = cust_id_big
        seg = str(cust.get("segment","")).strip().lower()
        seg_ranges = get_ranges_for_segment(seg)

        monthly_salary  = float(cust.get("monthly_salary_usd", 0.0))
        monthly_passive = float(cust.get("monthly_passive_income_usd", 0.0))
        monthly_rent_in = float(cust.get("monthly_rent_in_usd", 0.0))
        monthly_rent_out= float(cust.get("monthly_rent_out_usd", 0.0))
        annual_income   = float(cust.get("annual_income", 12*monthly_salary))

        bill_resp = parse_bool(cust.get("bill_responsible", False))
        is_renter = parse_bool(cust.get("is_renter", False))
        bills_due_day = due_day_if_flag(bill_resp, cust.get("bills_due_day"), default_day=10)
        rent_due_day  = due_day_if_flag(is_renter, cust.get("rent_due_day"),  default_day=5)
        salary_day = safe_day_from_cell(cust.get("monthly_salary_day", 1), default_day=1)

        weekly_rate = float(cust.get("weekly_txn_rate", 15.0))
        tod_mean = float(cust.get("tod_mean_hour", 13.0))
        tod_std  = float(cust.get("tod_std_hour", 3.0))

        current_balance = float(annual_income)/15.0
        cash_shadow = 0.0
        mu_week = np.random.normal(loc=weekly_rate, scale=max(0.01, weekly_rate*0.10))
        base_daily_rate = max(0.0, mu_week)/7.0

        # 1) Schedule recurring
        scheduled: Dict[date, List[dict]] = {}
        month_cursor = date(start.year, start.month, 1)
        while month_cursor <= end:
            y,m = month_cursor.year, month_cursor.month
            s_day = clamp_day(y,m,salary_day)
            if start<=s_day<=end and monthly_salary>0:
                _add_sched(scheduled, s_day, {"time": cfg.salary_time, "type_id": TT["salary"],
                                              "amount": round(monthly_salary,2), "merchant_id": None,
                                              "channel":"online", "is_recurring": True})
            if monthly_passive>0:
                p_day = clamp_day(y,m,20)
                if start<=p_day<=end:
                    _add_sched(scheduled, p_day, {"time": cfg.passive_time, "type_id": TT["passive_income"],
                                                  "amount": round(monthly_passive,2), "merchant_id": None,
                                                  "channel":"online", "is_recurring": True})
            if monthly_rent_in>0:
                rin_day = clamp_day(y,m,5)
                if start<=rin_day<=end:
                    _add_sched(scheduled, rin_day, {"time": cfg.rent_in_time, "type_id": TT["rent_in"],
                                                    "amount": round(monthly_rent_in,2), "merchant_id": None,
                                                    "channel":"online", "is_recurring": True})
            if rent_due_day is not None and monthly_rent_out>0:
                rout_day = clamp_day(y,m,rent_due_day)
                if start<=rout_day<=end:
                    _add_sched(scheduled, rout_day, {"time": cfg.rent_time, "type_id": TT["rent_out"],
                                                     "amount": round(monthly_rent_out,2), "merchant_id": None,
                                                     "channel":"online", "is_recurring": True})
            if bills_due_day is not None and monthly_salary>0:
                b_day = clamp_day(y,m,bills_due_day)
                if start<=b_day<=end:
                    for _,amt in _bill_amounts(monthly_salary):
                        _add_sched(scheduled, b_day, {"time": cfg.bills_time, "type_id": TT["bill_payment"],
                                                      "amount": round(amt,2), "merchant_id": 0,
                                                      "channel":"online", "is_recurring": True})
            month_cursor = date(y+(m//12),(m%12)+1,1)

        # 2) Build month caps & avg ticket
        credit_score = float(cust.get("credit_score", 650))
        # Compute discipline from age + education (using end of simulation as reference date)
        disc, age_years, edu_level, credit_norm = discipline_from_age_education_credit(
        cust.get("date_of_birth"),
        cust.get("education_level"),
        cust.get("credit_score"),
        cust.get("segment"),
        end  # reference date for age
        )

        daily_caps = build_monthly_caps(
        scheduled=scheduled,
        start=start, end=end,
        monthly_salary=monthly_salary,
        monthly_passive=monthly_passive,
        monthly_rent_in=monthly_rent_in,
        monthly_rent_out=monthly_rent_out,
        discipline=disc,   # << now driven by edu+credit
        )

        # (Optional) If you want to debug or export the derived level:
        print(cust["customer_id"], edu_level, disc)

        avg_ticket = expected_ticket_amount(cust, seg_ranges, monthly_salary)

        # 3) Day loop
        for d in all_days:
            cash_shadow *= cfg.cash_shadow_decay
            cash_shadow = min(cfg.cash_shadow_cap, max(0.0, cash_shadow))

            # Post recurring first
            if d in scheduled:
                for evt in sorted(scheduled[d], key=lambda e: e["time"]):
                    prev=round(current_balance,2); amt=float(evt["amount"])
                    current_balance += amt if evt["type_id"] in CREDIT_TYPES else -amt
                    newb=round(current_balance,2)
                    rows.append({
                        "transaction_id_dd": next_txn_id(seq),
                        "transaction_date": d.isoformat(),
                        "transaction_time": evt["time"].strftime("%H:%M:%S"),
                        "transaction_date_key": yyyymmdd_int(d),
                        "customer_id": cust_id_big,
                        "account_id": account_id_big,
                        "merchant_id": evt["merchant_id"],
                        "branch_id": int(branch_id),
                        "channel_id": _channel_id_for_name(channels_df, evt["channel"]),
                        "transaction_type_id": int(evt["type_id"]),
                        "amount": round(amt,2),
                        "currency": cfg.currency,
                        "is_recurring": True,
                        "previous_balance": prev,
                        "balance_after_transaction": newb,
                    }); seq+=1

            # Budget-aware variable spend
            thin = daily_thin_factor(current_balance, monthly_salary, cfg)
            cap_today = float(daily_caps.get(d, 0.0))
            cap_purchase_today = cap_today * (1.0 - cfg.cash_shadow_amount_weight * cash_shadow)
            base_lam = (base_daily_rate * 0.30) * (1.0 - cash_shadow) if cap_purchase_today < 1.0 else base_daily_rate * thin * (1.0 - cash_shadow)
            expected_count_cap = cap_purchase_today / max(1.0, avg_ticket)
            #lam = min(base_lam, max(0.0, expected_count_cap * 1.15))
            lam = base_lam
            k = np.random.poisson(lam=lam) if lam>0 else 0 #new fix

            atm_withdraw_total=0.0; var_spend_today=0.0; slack=0.15
            for _ in range(k):
                channel_id = pick_channel_id(cust, channels_df)
                channel_name = channel_name_by_id(channels_df, channel_id).lower()
                t = sample_time_from_normal(tod_mean, tod_std, min_hour=10.0)

                if channel_name == "atm":
                    ttype = TT["atm_withdrawal"]
                    rng = seg_ranges.get("atm", DEFAULT_CATEGORY_RANGES["atm"])
                    amt = round(max(5.0, amount_from_range(rng)*income_scale(monthly_salary)), 2)
                    merchant_id=None; atm_withdraw_total += amt
                else:
                    # ---- PURCHASE path (more variety) ----
                    ttype = TT["purchase"]

                    # pick category by per-customer weights
                    cats, ws = [], []
                    for suffix, cat in WEIGHT_COL_TO_CATEGORY.items():
                        if cat == "atm":
                            continue
                        col = f"w_cat_{suffix}"
                        if col in cust:
                            cats.append(cat)
                            ws.append(float(cust[col]))
                    category = weighted_choice(cats, ws) if cats else "shopping"

                    # pick merchant BEFORE amount, to apply merchant price index
                    merchant_id = choose_random_merchant(merchants_df, categories_df, category)
                    m_idx = merchant_price_index.get(int(merchant_id), 1.0) if merchant_id is not None else 1.0

                    # amount with tiered mix + person variance + merchant index + seasonality + shocks
                    amt = sample_purchase_amount(category, seg_ranges, disc, m_idx, d)

                    # Low-balance spend pressure shrinks purchase sizes a bit (kept from your model)
                    press = amount_spend_pressure(current_balance, monthly_salary, cfg)
                    amt = round(max(1.0, amt * press), 2)

                    # Respect today's purchase cap with SOFT logic (avoid constant “remaining” values)
                    if cap_purchase_today > 0:
                        remaining = cap_purchase_today * (1.0 + slack) - var_spend_today
                        keep, adj_amt = soft_cap(amt, remaining, slack=slack, discipline=disc)
                        if not keep:
                            # skip this purchase entirely
                            continue
                        amt = adj_amt

                    var_spend_today += amt


                prev=round(current_balance,2)
                current_balance += amt if ttype in CREDIT_TYPES else -amt
                newb=round(current_balance,2)

                rows.append({
                    "transaction_id_dd": next_txn_id(seq),
                    "transaction_date": d.isoformat(),
                    "transaction_time": t.strftime("%H:%M:%S"),
                    "transaction_date_key": yyyymmdd_int(d),
                    "customer_id": cust_id_big,
                    "account_id": account_id_big,
                    "merchant_id": int(merchant_id) if merchant_id is not None else None,
                    "branch_id": int(branch_id),
                    "channel_id": int(channel_id),
                    "transaction_type_id": int(ttype),
                    "amount": round(amt,2),
                    "currency": cfg.currency,
                    "is_recurring": False,
                    "previous_balance": prev,
                    "balance_after_transaction": newb,
                }); seq+=1

            if atm_withdraw_total>0 and monthly_salary>0:
                gain = atm_withdraw_total/(monthly_salary*cfg.cash_shadow_gain_div_salary)
                gain = max(0.10, min(cfg.cash_shadow_cap, gain))
                cash_shadow = min(cfg.cash_shadow_cap, cash_shadow + gain)

    out_cols = ["transaction_id_dd","transaction_date","transaction_time","transaction_date_key",
                "customer_id","account_id","merchant_id","branch_id","channel_id","transaction_type_id",
                "amount","currency","is_recurring","previous_balance","balance_after_transaction"]
    return pd.DataFrame(rows, columns=out_cols)

