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
                cap_today = float(daily_caps_q.get(d, 0.0))
                cap_purchase_today = cap_today * (1.0 - cfg.cash_shadow_amount_weight * cash_shadow)

                # counts & gentle amount jitter
                try:
                    count_f, amount_jitter = daily_thin_factors(current_balance, monthly_salary, d, disc_q, cash_shadow, cfg)
                except Exception:
                    thin = daily_thin_factor(current_balance, monthly_salary, cfg)
                    count_f, amount_jitter = thin, 1.0

                base_lam = base_daily_rate_q * count_f
                expected_count_cap = cap_purchase_today / max(1.0, avg_ticket)
                lam = min(base_lam, max(0.0, expected_count_cap * 1.15))
                k = int(np.random.poisson(lam=lam)) if lam > 0 else 0

                atm_withdraw_total = 0.0
                var_spend_today = 0.0
                slack = 0.15

                for _ in range(k):
                    # pick channel from quarter weights
                    try:
                        channel_id = pick_channel_with_override(ch_w_q, channels_df)
                    except Exception:
                        channel_id = pick_channel_id(cust, channels_df)
                    channel_name = channel_name_by_id(channels_df, channel_id).lower()

                    # time of day (keep salary first visually)
                    t = sample_time_from_normal(tod_mean, tod_std, min_hour=10.0)

                    if channel_name == "atm":
                        # ATM path → use your ATM merchants
                        ttype = TT["atm_withdrawal"]
                        rng = seg_ranges.get("atm", DEFAULT_CATEGORY_RANGES["atm"])
                        amt = amount_from_range(rng) * income_scale(monthly_salary) * price_level
                        amt = round(max(5.0, amt), 2)
                        merchant_id = pick_merchant_id_for_category("atm")
                        if not isinstance(merchant_id, (int, np.integer)):
                            merchant_id = 0
                        atm_withdraw_total += amt
                        is_salary_flag = False

                    else:
                        # PURCHASE path → category merchant by category
                        ttype = TT["purchase"]

                        try:
                            category = pick_category_with_override(cat_w_q)
                        except Exception:
                            cats, ws = [], []
                            for suffix, cat in WEIGHT_COL_TO_CATEGORY.items():
                                if cat == "atm":
                                    continue
                                col = f"w_cat_{suffix}"
                                if col in cust:
                                    cats.append(cat)
                                    ws.append(float(cust[col]))
                            category = weighted_choice(cats, ws) if cats else "shopping"

                        merchant_id = choose_random_merchant(merchants_df, categories_df, category)
                        if merchant_id is None:
                            merchant_id = 0
                        else:
                            merchant_id = int(merchant_id)

                        m_idx = merchant_price_index.get(int(merchant_id), 1.0) if merchant_id is not None else 1.0

                        # amount (sampler → press → jitter → price level)
                        try:
                            amt = sample_purchase_amount(category, seg_ranges, disc_q, m_idx, d)
                        except Exception:
                            lo, hi = seg_ranges.get(category, DEFAULT_CATEGORY_RANGES.get(category, (10, 120)))
                            base_amt = amount_from_range((lo, hi)) * income_scale(monthly_salary)
                            amt = max(1.0, base_amt * m_idx)

                        press = amount_spend_pressure(current_balance, monthly_salary, cfg)
                        amt = max(1.0, amt * press * amount_jitter * price_level)

                        # soft cap
                        if cap_purchase_today > 0:
                            remaining = cap_purchase_today * (1.0 + slack) - var_spend_today
                            try:
                                keep, adj_amt = soft_cap(amt, remaining, slack=slack, discipline=disc_q)
                            except Exception:
                                keep, adj_amt = (remaining > 2.0, min(amt, max(1.0, remaining)))
                            if not keep:
                                continue
                            amt = adj_amt
                        var_spend_today += amt
                        amt = round(amt, 2)
                        is_salary_flag = False

                    # post & record
                    prev = round(current_balance, 2)
                    if ttype in CREDIT_TYPES:
                        current_balance += amt
                    else:
                        current_balance -= amt
                    newb = round(current_balance, 2)

                    rows.append({
                        "transaction_id_dd": next_txn_id(seq),
                        "transaction_date": d.isoformat(),
                        "transaction_time": t.strftime("%H:%M:%S"),
                        "transaction_date_key": yyyymmdd_int(d),
                        "customer_id": cust_id_big,
                        "account_id": account_id_big,
                        "merchant_id": int(merchant_id) if merchant_id is not None else 0,
                        "branch_id": int(branch_id),
                        "channel_id": int(channel_id),
                        "transaction_type_id": int(ttype),
                        "amount": round(amt, 2),
                        "currency": cfg.currency,
                        "is_recurring": False,
                        "is_salary": bool(is_salary_flag),
                        "previous_balance": prev,
                        "balance_after_transaction": newb,
                    })
                    seq += 1
                    update_qmetrics(qm, newb, ttype, amt)

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