import pandas as pd
import logging

logger = logging.getLogger(__name__)

class RuleEngine:

    OUR_BANK_ACCOUNTS = ['5620167105','1726072612', '2000561656', 
                     '1012834872','2001579632','0786783052','1025470643']
    
    OUR_BANK_NAME = 'our_bank_name'
    OUR_ORGANIZATION_NAME = 'our_organization_name'
    WALLET_LIMIT = {"Zedvance Pro Wallet": 1_000_000,"Business Wallet": 10_000_000}

    def __init__(self,df,trans_df,users_df):
        self.df = df
        self.trans_df = trans_df
        self.users_df = users_df
        self.succesful = (self.df['status'] == 'successful')
        self.succesful_outflow_mask = self.succesful & (self.df['category'] == 'Outflow')
        self.loan_mask = self.df['category'].str.lower().isin(['salary loan', 'staff loan', 'business loans'])
        self.repayment = (self.df['category'].str.lower() == 'loan repayment')
        self.successful_inflow = (self.df['category'] == 'Inflow') & self.succesful

    def flag_negative_balance(self):
        logging.info('Running negative balance check...')
        self.df['negative_balance_flag'] = self.df['Total Balance (Transaction)'] < 0
        result = self.df.groupby('Account ID (Transaction)')['negative_balance_flag'].any().reset_index()
    
        logging.info(f"Check completed. Flagged {len(result[result['negative_balance_flag']==True])} customers with suspected negtive balance.")
        return result

    def flag_round_figure(self, threshold=5, min_amount=100_000):
        logging.info("Running round-figure check (transaction-aware)...")

        # Initialize transaction-level flag
        self.df['round_figure_flag'] = False

        # Work on filtered view, keep original index
        df = (
            self.df
            .loc[
                self.succesful_outflow_mask &
                (self.df['amount'] >= min_amount)
            ]
            .assign(txn_idx=lambda x: x.index)  # preserve original index
            .copy()
        )

        # Identify round-figure transactions
        df['is_round_figure'] = df['amount'] % 1_000 == 0

        # Count round-figure transactions per account per day
        daily_counts = (
            df[df['is_round_figure']]
            .groupby(['Account ID (Transaction)', 'txn_date'])
            .size()
            .reset_index(name='round_figure_count')
        )

        # Identify days exceeding threshold
        flagged_days = daily_counts[daily_counts['round_figure_count'] > threshold]

        # Merge back to identify the transactions to flag
        flagged_txns = df.merge(
            flagged_days[['Account ID (Transaction)', 'txn_date']],
            on=['Account ID (Transaction)', 'txn_date'],
            how='inner'
        ).query('is_round_figure')

        # Flag original transactions using preserved index
        self.df.loc[flagged_txns['txn_idx'], 'round_figure_flag'] = True

        # Account-level aggregation
        account_flags = (
            self.df
            .groupby('Account ID (Transaction)')['round_figure_flag']
            .any()
            .reset_index()
        )

        logging.info(
            f"Check completed. Flagged {account_flags['round_figure_flag'].sum()} customers "
            "with high number of round-figure transactions."
        )

        return account_flags

    def flag_odd_hour(self):
        logging.info('Running odd hours check...')
        # identify the transactions with suspicions
        self.df['flag_odd_hour'] = (
        self.succesful_outflow_mask &
        (self.df['txn_hour'] >= 0) &  # midnight included
        (self.df['txn_hour'] < 5)    # early morning
    )

        # identify the users with suspicions
        result = self.df.groupby('Account ID (Transaction)')['flag_odd_hour'].any().reset_index()
        
        logging.info(f"Check completed. Flagged {result['flag_odd_hour'].sum()} customers with odd hour transactions.")
        return result

    def flag_high_transaction_velocity(
        self,
        time_window_minutes=5,
        threshold=5,
        min_amount = 1_000_000
    ):
        logging.info('Running transaction velocity check (transaction-aware)...')

        # Initialize column (idempotent)
        self.df['flag_high_velocity'] = False

        df = (
            self.df[
                self.succesful_outflow_mask &
                (self.df['amount'] >= min_amount)
            ]
            .sort_values(['Account ID (Transaction)', 'Booking Date (Entry Date)'])
        )

        window = pd.Timedelta(minutes=time_window_minutes)

        for acct, group in df.groupby('Account ID (Transaction)', sort=False):
            times = group['Booking Date (Entry Date)'].to_numpy()
            idx = group.index.to_numpy()

            start = 0
            for end in range(len(times)):
                while times[end] - times[start] > window:
                    start += 1

                if (end - start + 1) > threshold:
                    # Flag transactions in this burst
                    self.df.loc[idx[start:end + 1], 'flag_high_velocity'] = True
                    # Break if you only care about first burst per account
                    break

        # Roll up to account level
        account_flags = (
            self.df
            .groupby('Account ID (Transaction)', as_index=False)['flag_high_velocity']
            .any()
        )

        flagged_count = account_flags['flag_high_velocity'].sum()
        logging.info(
            f"Check completed. Flagged {flagged_count} customers who "
            f"performed ≥{threshold} transactions within {time_window_minutes} minutes"
        )

        return account_flags

    def flag_structured_outflow(
        self,
        min_amt=1_000_000,
        max_amt=5_000_000,
        threshold_total=10_000_000,
        window_days=7
    ):
        logging.info('Running structured outflow check (same-receiver, transaction-aware)...')

        # Initialize column
        self.df['weekly_flag_structuring'] = False

        # Filter relevant transactions
        df = (
            self.df[
                self.succesful_outflow_mask &
                (self.df['amount'].between(min_amt, max_amt))
            ]
            .sort_values(['Account ID (Transaction)', 'Booking Date (Entry Date)'])
        )

        window = pd.Timedelta(days=window_days)

        # Process per account
        for acct, group in df.groupby('Account ID (Transaction)', sort=False):

            times = group['Booking Date (Entry Date)'].to_numpy()
            amounts = group['amount'].to_numpy()
            receivers = group['Receiver_Account'].astype(str).to_numpy()
            idx = group.index.to_numpy()

            start = 0
            running_sum = 0

            for end in range(len(times)):
                running_sum += amounts[end]

                while times[end] - times[start] > window:
                    running_sum -= amounts[start]
                    start += 1

                # ---- SAME RECEIVER CONDITION ----
                window_receivers = receivers[start:end + 1]
                same_receiver = len(set(window_receivers)) == 1

                # ---- STRUCTURING CONDITIONS ----
                if running_sum >= threshold_total and same_receiver:
                    self.df.loc[
                        idx[start:end + 1],
                        'weekly_flag_structuring'
                    ] = True
                    break  # flag once per account

        # Roll up to account level
        account_flags = (
            self.df
            .groupby('Account ID (Transaction)', as_index=False)['weekly_flag_structuring']
            .any()
        )

        flagged_count = account_flags['weekly_flag_structuring'].sum()
        logging.info(
            f"Check completed. Flagged {flagged_count} customers with suspected same-receiver outflow structuring."
        )

        return account_flags

    def flag_early_loan_repayment_per_disbursement(self, days_threshold=7, min_repay_ratio=0.5):
        logging.info("Running early loan repayment check (transaction-aware)...")

        # Do not copy here if you want to modify original df in-place
        self.df['flag_early_repayment'] = False

        # Keep only relevant transactions and preserve index
        disb = (
            self.df[
                self.loan_mask
            ]
            .assign(disb_idx=lambda x: x.index)
            [['Account ID (Transaction)', 'Booking Date (Entry Date)', 'amount', 'disb_idx']]
            .rename(columns={'Booking Date (Entry Date)': 'disb_date', 'amount': 'disb_amount'})
        )

        repay = (
            self.df[
                self.repayment
            ]
            .assign(repay_idx=lambda x: x.index)
            [['Account ID (Transaction)', 'Booking Date (Entry Date)', 'amount', 'repay_idx']]
            .rename(columns={'Booking Date (Entry Date)': 'repay_date', 'amount': 'repay_amount'})
        )

        # Process per customer to avoid cartesian multiplication
        flagged_repay_idx = []
        flagged_disb_idx = []

        for cust_id, disb_group in disb.groupby('Account ID (Transaction)'):
            repay_group = repay[repay['Account ID (Transaction)'] == cust_id]

            for _, disb_row in disb_group.iterrows():
                # repayments after disbursement within threshold
                mask = (
                    (repay_group['repay_date'] > disb_row['disb_date']) &
                    (repay_group['repay_date'] <= disb_row['disb_date'] + pd.Timedelta(days=days_threshold)) &
                    (repay_group['repay_amount'] >= min_repay_ratio * disb_row['disb_amount']) &
                    (repay_group['repay_amount'] <= disb_row['disb_amount'])
                )
                flagged = repay_group[mask]
                if not flagged.empty:
                    flagged_repay_idx.extend(flagged['repay_idx'].tolist())
                    flagged_disb_idx.append(disb_row['disb_idx'])

        # Flag transactions in the original dataframe
        self.df.loc[flagged_repay_idx, 'flag_early_repayment'] = True
        self.df.loc[flagged_disb_idx, 'flag_early_repayment'] = True

        # Account-level summary
        account_flags = (
            self.df.groupby('Account ID (Transaction)', as_index=False)['flag_early_repayment'].any()
        )

        logging.info(
            f"Check completed. Flagged {account_flags['flag_early_repayment'].sum()} customers with suspected unusual repayment behavior"
        )

        return account_flags

    def flag_third_party_repayments(self):
        logging.info("Running third-party repayment flag check (transaction-aware)...")

        # Initialize column
        self.df['flag_third_party_repayment'] = False

        # Ensure Sender_Bank is string
        sender_bank = self.df.loc[self.repayment, 'Sender_Bank'].astype(str)

        # Identify third-party repayments
        third_party_mask = ~sender_bank.str.lower().str.contains(self.OUR_BANK_NAME.lower())

        # Flag the actual repayment transactions
        self.df.loc[
            self.repayment & third_party_mask,
            'flag_third_party_repayment'
        ] = True

        # Roll up to account level
        account_flags = (
            self.df
            .groupby('Account ID (Transaction)', as_index=False)['flag_third_party_repayment']
            .any()
        )

        flagged_count = account_flags['flag_third_party_repayment'].sum()
        logging.info(
            f"Check completed. Flagged {flagged_count} customers with suspected 3rd-party repayments."
        )

        return account_flags

    def flag_linked_accounts_by_bvn(self):
        logging.info(f'Running duplicate bvn check...')

        user_data = self.users_df.copy()
        # Count distinct accounts per ID (across all products)
        acct_counts = user_data.groupby('BVN')['Account ID (Transaction)'].transform('nunique')

        # Flag in-place
        user_data['linked_bvn_flag'] = acct_counts > 1

        total_flagged = user_data['linked_bvn_flag'].sum()

        logging.info(f"Check completed. Flagged {total_flagged} customer accounts with multiple accounts.")

        return user_data[['Account ID (Transaction)', 'linked_bvn_flag']].drop_duplicates()

    def flag_transaction_surge(self,
        historical_days=90,
        multiplier=10,
        method="hybrid",     # "multiplier", "zscore", "percentile", "hybrid"
        z_threshold=4,
        percentile=0.99,min_amount=10_000
    ):
        logging.info("Running transaction surge check (enhanced, transaction-aware)...")

        # -----------------------------
        # 1. PREPARE HISTORICAL DATA
        # -----------------------------
        hist = self.trans_df[
            (self.trans_df['category'] == 'Outflow') & 
            (self.trans_df['status'] == 'successful')
        ].copy()

        current_end = self.df['Booking Date (Entry Date)'].max()
        recent_start = self.df['Booking Date (Entry Date)'].min()
        recent_days = max((current_end - recent_start).days, 1)

        historical_start = current_end - pd.Timedelta(days=historical_days + recent_days)

        hist['period'] = 'ignore'
        hist.loc[
            (hist['Booking Date (Entry Date)'] >= historical_start) &
            (hist['Booking Date (Entry Date)'] < recent_start),
            'period'
        ] = 'historical'

        hist.loc[
            hist['Booking Date (Entry Date)'] >= recent_start,
            'period'
        ] = 'recent'

        hist = hist[hist['period'] != 'ignore']

        # -----------------------------
        # 2. DAILY AGGREGATION
        # -----------------------------
        daily = (
            hist
            .groupby(['Account ID (Transaction)', 'period', 'txn_date'])
            .agg(
                daily_count=('amount', 'count'),
                daily_value=('amount', 'sum')
            )
            .reset_index()
        )

        # -----------------------------
        # 3. HISTORICAL DISTRIBUTION
        # -----------------------------
        hist_dist = (
            daily[daily['period'] == 'historical']
            .groupby('Account ID (Transaction)')
            .agg(
                mean_count=('daily_count', 'mean'),
                std_count=('daily_count', 'std'),
                mean_value=('daily_value', 'mean'),
                std_value=('daily_value', 'std'),
                pctl_count=('daily_count', lambda x: x.quantile(percentile)),
                pctl_value=('daily_value', lambda x: x.quantile(percentile))
            )
            .reset_index()
            .fillna(0)
        )

        # -----------------------------
        # 4. RECENT DAILY METRICS
        # -----------------------------
        recent = daily[daily['period'] == 'recent']

        merged = recent.merge(
            hist_dist,
            on='Account ID (Transaction)',
            how='left'
        ).fillna(0)

        # -----------------------------
        # 5. SURGE METRICS
        # -----------------------------
        merged['z_count'] = (
            (merged['daily_count'] - merged['mean_count']) /
            merged['std_count'].replace(0, 1)
        )

        merged['z_value'] = (
            (merged['daily_value'] - merged['mean_value']) /
            merged['std_value'].replace(0, 1)
        )

        # -----------------------------
        # 6. SURGE DECISION
        # -----------------------------
        if method == "multiplier":
            merged['flag_txn_surge'] = (
                (merged['daily_count'] > multiplier * merged['mean_count']) &
                (merged['daily_value'] > multiplier * merged['mean_value'])
            )

        elif method == "zscore":
            merged['flag_txn_surge'] = (
                (merged['z_count'] >= z_threshold) &
                (merged['z_value'] >= z_threshold)
            )

        elif method == "percentile":
            merged['flag_txn_surge'] = (
                (merged['daily_count'] >= merged['pctl_count']) &
                (merged['daily_value'] >= merged['pctl_value'])
            )

        elif method == "hybrid":
            merged['flag_txn_surge'] = (
                (merged['daily_count'] > multiplier * merged['mean_count']) &
                (merged['z_count'] >= z_threshold) &
                (merged['daily_value'] > multiplier * merged['mean_value']) &
                (merged['z_value'] >= z_threshold)
            )

        else:
            raise ValueError("Invalid surge detection method")
        
        merged['flag_txn_surge'] &= (merged['daily_value'] > min_amount)
        # -----------------------------
        # 7. FLAG ONLY SURGE DAYS (transaction-aware)
        # -----------------------------
        flagged_days = (
            merged.loc[
                merged['flag_txn_surge'],
                ['Account ID (Transaction)', 'txn_date']
            ]
            .drop_duplicates()
        )

        flag_map = (
            flagged_days
            .assign(flag_txn_surge=True)
            .set_index(['Account ID (Transaction)', 'txn_date'])['flag_txn_surge']
        )

        # Initialize column for all transactions
        self.df['flag_txn_surge'] = False

        # Only apply mapping to Outflow transactions
        outflow_mask = self.df['category'] == 'Outflow'

        self.df.loc[outflow_mask, 'flag_txn_surge'] = (
            self.df.loc[outflow_mask]
            .set_index(['Account ID (Transaction)', 'txn_date'])
            .index
            .map(flag_map)
            .fillna(False)
        )

        # -----------------------------
        # 8. CUSTOMER-LEVEL RETURN
        # -----------------------------
        customer_flags = (
            self.df
            .groupby('Account ID (Transaction)')['flag_txn_surge']
            .any()
            .reset_index()
        )

        logging.info(
            f"Check completed. Flagged {customer_flags['flag_txn_surge'].sum()} customers "
            f"with suspected transaction surge, using {method} surge detection."
        )

        if customer_flags.empty:
            return (
                self.df[['Account ID (Transaction)']]
                .drop_duplicates()
                .assign(flag_txn_surge=False)
            )

        return customer_flags

    def single_trans_check(self):
        logging.info("Running single transaction value check (transaction-aware)...")

        # Initialize flag column
        self.df['single_trans_flag'] = False

        # Filter relevant transactions
        df = self.df[
            self.succesful_outflow_mask
        ].copy()

        # Map wallet limits
        df['limit'] = df['GL Account Name'].map(self.WALLET_LIMIT)

        # Identify violating transactions
        violating_idx = df[df['amount'] > df['limit']].index

        # Flag ONLY those transactions
        self.df.loc[violating_idx, 'single_trans_flag'] = True

        # Account-level aggregation
        account_flags = (
            self.df
            .groupby('Account ID (Transaction)')['single_trans_flag']
            .any()
            .reset_index(name='single_trans_flag')
        )

        flagged_count = account_flags['single_trans_flag'].sum()
        logging.info(
            f"Check completed. Flagged {flagged_count} customers with exceeded single transaction limit"
        )

        return account_flags
    
    def flag_large_inflows_on_new_accounts(self, hours_threshold=48, inflow_threshold=50_000_000):
        logging.info("Running new customer inflow check (transaction-aware)...")

        # Initialize flag
        self.df['flag_early_50m_inflow'] = False

        # Relevant inflow transactions
        df = self.df[
            self.successful_inflow & 
            (self.df['Sender_Name'] != self.OUR_ORGANIZATION_NAME)
        ].copy()

        # Hours since account opening
        df['hours_since_opening'] = (
            df['Booking Date (Entry Date)'] - df['Created']
        ).dt.total_seconds() / 3600

        # Only inflows within threshold window
        df = df[df['hours_since_opening'] <= hours_threshold]

        # Process per account
        for acct, group in df.groupby('Account ID (Transaction)', sort=False):
            total_inflow = group['amount'].sum()

            if total_inflow >= inflow_threshold:
                # Flag ONLY contributing transactions
                self.df.loc[group.index, 'flag_early_50m_inflow'] = True

        # Account-level aggregation
        account_flags = (
            self.df
            .groupby('Account ID (Transaction)')['flag_early_50m_inflow']
            .any()
            .reset_index(name='flag_early_50m_inflow')
        )

        flagged_count = account_flags['flag_early_50m_inflow'].sum()
        logging.info(
            f"Check completed. Flagged {flagged_count} new customers with ≥ 50M inflow within {hours_threshold} hours"
        )

        return account_flags


    def flag_outflow_on_new_accounts(self,hours_threshold=24):
        logging.info(f'Running new customer outflow check...')

        # Filter outflows(>=50k) that occurred within 24 hours of account opening
        self.df['hours_since_opening'] = (self.df['Booking Date (Entry Date)'] - self.df['Created']).dt.total_seconds() / 3600
        self.df['new_account_outflow_flag'] = ((self.df['hours_since_opening'] <= hours_threshold) &
                                                        self.succesful_outflow_mask &
                                                        self.df['amount']>=50_000
                                                        )

        new_customer_outflows = (
        self.df
        .groupby('Account ID (Transaction)')['new_account_outflow_flag']
        .any()
        .reset_index())

        logging.info(f"Check completed. Flagged {new_customer_outflows['new_account_outflow_flag'].sum()} new customers with at least one outflow in the first 24hrs")
        
        self.df.drop(['hours_since_opening'],axis=1, inplace=True)

        return new_customer_outflows

    def flag_rapid_withdrawals_by_customer(
        self,
        inflow_threshold=5_000_000,
        window_minutes=5,
        similarity_ratio=0.9  # 90%+ of inflow considered similar
    ):
        logging.info('Running rapid withdrawals check (amount-aware)...')

        self.df['flag_rapid_turnover'] = False
        window = pd.Timedelta(minutes=window_minutes)

        # Inflows (keep index + amount)
        inflows_df = self.df[
            self.successful_inflow &
            (self.df['amount'] >= inflow_threshold)
        ][
            ['Account ID (Transaction)', 'Booking Date (Entry Date)', 'amount']
        ]

        # Outflows (keep index + amount)
        outflows_df = self.df[
            self.succesful_outflow_mask
        ][
            ['Account ID (Transaction)', 'Booking Date (Entry Date)', 'amount']
        ]

        # Process per account
        for acct, inflows_group in inflows_df.groupby('Account ID (Transaction)'):

            outflows_group = outflows_df[
                outflows_df['Account ID (Transaction)'] == acct
            ]

            if outflows_group.empty:
                continue

            out_idx = outflows_group.index.to_numpy()
            out_times = outflows_group['Booking Date (Entry Date)'].to_numpy()
            out_amounts = outflows_group['amount'].to_numpy()

            for inflow_idx, inflow in inflows_group.iterrows():

                inflow_time = inflow['Booking Date (Entry Date)']
                inflow_amount = inflow['amount']

                mask = (out_times > inflow_time) & (out_times <= inflow_time + window)

                if not mask.any():
                    continue

                total_withdrawn = out_amounts[mask].sum()

                # Check similarity (same or similar amount)
                if total_withdrawn >= similarity_ratio * inflow_amount:
                    # Flag inflow
                    self.df.loc[inflow_idx, 'flag_rapid_turnover'] = True

                    # Flag only related outflows
                    self.df.loc[out_idx[mask], 'flag_rapid_turnover'] = True

        # Account-level aggregation
        account_flags = (
            self.df
            .groupby('Account ID (Transaction)')['flag_rapid_turnover']
            .any()
            .reset_index()
        )

        logging.info(
            f"Check completed. Flagged {account_flags['flag_rapid_turnover'].sum()} customers "
            f"with rapid turnover where withdrawals ≥ {int(similarity_ratio*100)}% of inflow within {window_minutes} minutes."
        )

        return account_flags

    def flag_duplicate_outflow(self,time_window_minutes=5,threshold=3):
        logging.info('Running duplicate outflow check...')

        # Initialize flag in-place
        self.df['flag_duplicate_debit'] = False

        # Work only on relevant transactions
        df = self.df[
            self.succesful_outflow_mask
        ]

        df = df.sort_values(['Account ID (Transaction)', 'Receiver_Account', 'amount', 'Booking Date (Entry Date)'])

        grouped = df.groupby(['Account ID (Transaction)', 'Receiver_Account', 'amount'],sort=False)

        for (cust_id, receiver, amount), group in grouped:
            times = group['Booking Date (Entry Date)'].reset_index(drop=True)

            for i in range(len(times)):
                window_end = times[i] + pd.Timedelta(minutes=time_window_minutes)

                mask_window = (times >= times[i]) & (times <= window_end)

                if mask_window.sum() >= threshold:
                    # Identify the actual rows to flag (IN PLACE)
                    idx_to_flag = group.loc[mask_window.values].index

                    self.df.loc[idx_to_flag, 'flag_duplicate_debit'] = True

                    # break to avoid over-flagging same pattern repeatedly
                    break

        # Customer-level result
        flagged_customers = (
            self.df
            .groupby('Account ID (Transaction)')['flag_duplicate_debit']
            .any()
            .reset_index(name='flag_duplicate_debit')
        )

        logging.info(
            f"Check completed. Flagged {flagged_customers['flag_duplicate_debit'].sum()} customers "
            f"with duplicate debit transactions."
        )

        return flagged_customers

    def flag_structured_large_inflow(self,threshold=5):
        logging.info(f'Running inflow high value spikes check...')

        # Filter only credits
        credit_txns = self.df[self.successful_inflow].copy()

        # Apply threshold filtering based on wallet_type
        credit_txns['limit'] = credit_txns["GL Account Name"].map(self.WALLET_LIMIT)
        filtered_inflow = credit_txns[credit_txns['amount'] >= credit_txns['limit']]
        
        # Sort
        filtered_inflow = filtered_inflow.sort_values(['Account ID (Transaction)', 'Booking Date (Entry Date)'])
        self.df['flag_high_value_spike_inflows'] = False

        # Group by customer
        for cust_id, group in filtered_inflow.groupby('Account ID (Transaction)'):
            times = group['Booking Date (Entry Date)'].reset_index(drop=True)

            for i in range(len(times)):
                    
                window_end = times[i] + pd.Timedelta(hours=24)

                mask_window = (times >= times[i]) & (times <= window_end)

                if mask_window.sum() >= threshold:
                    # Identify the actual rows to flag (IN PLACE)
                    idx_to_flag = group.loc[mask_window.values].index

                    self.df.loc[idx_to_flag, 'flag_high_value_spike_inflows'] = True

                    # break to avoid over-flagging same pattern repeatedly
                    break

        # Customer-level result
        flagged_customers = (
            self.df
            .groupby('Account ID (Transaction)')['flag_high_value_spike_inflows']
            .any()
            .reset_index()
        )
        logging.info(f"Check completed. Flagged {flagged_customers['flag_high_value_spike_inflows'].sum()} customers with 5 or more large inflows that totals at least (single:1M, SME:5M) within 24 hours")

        return flagged_customers

    def flag_structured_transfers(self,time_window_minutes=60,min_txn_count=3):
        logging.info('Running structured transfer check...')

        self.df['flag_structured_transfer'] = False

        df = self.df[
            self.succesful_outflow_mask
        ].sort_values(
            ['Account ID (Transaction)', 'Receiver_Account', 'Booking Date (Entry Date)']
        )

        grouped = df.groupby(
            ['Account ID (Transaction)', 'Receiver_Account'],
            sort=False
        )

        for (cust_id, receiver), group in grouped:
            times = group['Booking Date (Entry Date)'].reset_index(drop=True)
            amounts = group['amount'].reset_index(drop=True)
            
            account_type = group["GL Account Name"].iloc[0]

            limit = self.WALLET_LIMIT.get(account_type)

            if limit is None:
                continue  # skip unknown wallet types


            for i in range(len(times)):
                window_end = times[i] + pd.Timedelta(minutes=time_window_minutes)

                mask_window = (times >= times[i]) & (times <= window_end)

                if (
                    mask_window.sum() >= min_txn_count and
                    amounts[mask_window].sum() >= limit
                ):
                    idx_to_flag = group.loc[mask_window.values].index

                    self.df.loc[idx_to_flag, 'flag_structured_transfer'] = True

                    # Stop after first detection per receiver
                    break

        result = (
            self.df
            .groupby('Account ID (Transaction)')['flag_structured_transfer']
            .any()
            .reset_index(name='flag_structured_transfer')
        )

        logging.info(
            f"Check completed. Flagged {result['flag_structured_transfer'].sum()} customers with  {min_txn_count} or more outflows  that totals at least (single:1M, SME:5M) within {time_window_minutes} minutes."
        )

        return result

    def active_dormant_check(self, threshold=180):
        logging.info('Running active dormant account check...')

        df = self.trans_df[['Account ID (Transaction)', 'txn_date']].copy()

        latest_date = self.df['Booking Date (Entry Date)'].min()


        # Get two most recent transactions per customer
        recent = (
            df
            .sort_values('txn_date', ascending=False)
            .groupby('Account ID (Transaction)')
            .head(2)
        )

        # Pivot to wide format
        recent_wide = (
            recent
            .assign(rank=recent.groupby('Account ID (Transaction)').cumcount())
            .pivot(
                index='Account ID (Transaction)',
                columns='rank',
                values='txn_date'
            )
            .rename(columns={0: 'max_date', 1: 'second_max_date'})
            .reset_index()
        )

        # Compute gap
        recent_wide['max_date'] = pd.to_datetime(recent_wide['max_date'], errors='coerce')
        recent_wide['second_max_date'] = pd.to_datetime(recent_wide['second_max_date'], errors='coerce')

        recent_wide['days_gap'] = (recent_wide['max_date'] - recent_wide['second_max_date']).dt.days

        # Flag dormant - active behavior
        recent_wide['dormant_activity_flag'] = (recent_wide['days_gap'] > threshold) & (recent_wide['max_date'] >= latest_date)

        logging.info(
            f"Check completed. Flagged {recent_wide['dormant_activity_flag'].sum()} accounts "
            f"with activity after {threshold}+ days of dormancy."
        )

        return recent_wide[['Account ID (Transaction)', 'dormant_activity_flag']]


    def tier1_daily_limit(self, threshold=50_000):
        logging.info(f'Running Tier-1 daily check...')

        # Filter transactions that occurred for Tier 1 accounts

        tier1_data = self.df[(self.df['Tier']==1) & 
                                    (self.df['ledger_type']=='debit') &
                                    (self.df['status']=='successful')].copy()
        
        # Sum daily transactions per customer
        daily_totals = tier1_data.groupby(['Account ID (Transaction)', 'txn_date'])['amount'].sum().reset_index(name='daily_total')

        # Flag if daily total exceeds limit
        daily_totals['tier1_daily_limit_flag'] = daily_totals['daily_total'] > threshold

        flag_map = (
            daily_totals
            .set_index(['Account ID (Transaction)', 'txn_date'])['tier1_daily_limit_flag']
        )

        self.df['tier1_daily_limit_flag'] = (
            self.df
            .set_index(['Account ID (Transaction)', 'txn_date'])
            .index
            .map(flag_map)
            .fillna(False)
        )

        # Return one row per customer with True/False flag
        customer_flags = (
            daily_totals
            .groupby('Account ID (Transaction)')['tier1_daily_limit_flag']
            .any()  
            .reset_index())
        logging.info(f"Check completed. Flagged {customer_flags['tier1_daily_limit_flag'].sum()} tier-1 customers who exceeded their daily limit")

        return customer_flags

    def tier1_balance_limit(self,threshold=300_000):
        logging.info(f'Running Tier-1 balance check...')

        # Filter transactions that occurred for Tier 1 accounts
        self.df['tier1_balance_limit_flag']=(self.df['Tier']==1) & (self.df['Total Balance (Transaction)'] > threshold) & (self.df['status']=='successful')

        # Return one row per customer with True/False flag
        customer_flags = (
            self.df
            .groupby('Account ID (Transaction)')['tier1_balance_limit_flag']
            .any()  # True if any flagged transaction
            .reset_index())
        logging.info(f"Check completed. Flagged {customer_flags['tier1_balance_limit_flag'].sum()} tier-1 customers who have above tier-1 balance limit")

        return customer_flags

    def flag_linked_transfers(self,time_window='1D'):
        logging.info('Running structured linked transfers check...')

        # Initialize flag IN PLACE
        self.df['flag_structured_linked_outflow'] = False

        # Identify BVNs with multiple accounts
        multi_account_bvns = (
            self.users_df
            .groupby(['BVN'])['Account ID (Transaction)']
            .nunique()
            .reset_index(name='acct_count')
        )

        multi_account_bvns = multi_account_bvns[
            multi_account_bvns['acct_count'] >= 2
        ]['BVN']

        # Filter customer data
        filtered_customer_df = self.users_df[
            self.users_df['BVN'].isin(multi_account_bvns)
        ]

        # Attach BVN to transactions
        txns = self.df.merge(
            filtered_customer_df[['Account ID (Transaction)', 'BVN']],
            on='Account ID (Transaction)',
            how='inner'
        )

        flagged_txn_idx = set()

        # Process per BVN
        for bvn, accounts in filtered_customer_df.groupby('BVN')['Account ID (Transaction)']:
            linked_accounts = set(accounts)

            group_txns = (
                txns[txns['Account ID (Transaction)'].isin(linked_accounts)]
                .sort_values('Booking Date (Entry Date)')
            )

            inflows = group_txns[group_txns['category'] == 'Inflow']

            for _, deposit in inflows.iterrows():
                start_time = deposit['Booking Date (Entry Date)']
                end_time = start_time + pd.Timedelta(time_window)

                post_txns = group_txns[
                    (group_txns['Booking Date (Entry Date)'] > start_time) &
                    (group_txns['Booking Date (Entry Date)'] <= end_time) &
                    (group_txns['category'] == 'Outflow')
                ]

                receiver_accounts = (
                    set(post_txns['Receiver_Account']) & linked_accounts
                )

                # Structured linked transfer detected
                if len(receiver_accounts) >= 2:
                    flagged_txn_idx.add(deposit.name)

                    outflow_idx = post_txns[
                        post_txns['Receiver_Account'].isin(linked_accounts)
                    ].index

                    flagged_txn_idx.update(outflow_idx)
                    break

        # Apply flags IN PLACE
        self.df.loc[
            self.df.index.isin(flagged_txn_idx),
            'flag_structured_linked_outflow'
        ] = True

        # Account-level summary
        summary = (
            self.df
            .groupby('Account ID (Transaction)')['flag_structured_linked_outflow']
            .any()
            .reset_index()
        )

        logging.info(
            f"Check completed. Flagged {summary['flag_structured_linked_outflow'].sum()} "
            "accounts with structured linked transfers"
        )

        return summary

    def flag_suspicious_rapid_cashout(self,min_amount=5_000_000,similarity_score = 0.9):
        """
        Flags suspicious rapid cashout transactions:
        - Tier 3 accounts (new or recently upgraded)
        - Inflows > 5,000,000
        - Followed by any outflow within 24 hours
        - Excludes certain sender accounts
        """

        logging.info('Running suspicious rapid cashout check...')

        # Ensure flag column exists
        self.df['flag_suspicious_rapid_cashout'] = False
        flagged_txn_idx = set()

        latest_date = self.df['Booking Date (Entry Date)'].max()
        recent_threshold = latest_date - pd.Timedelta(days=30)

        # Step 1: Filter qualifying inflows
        inflows = self.df[
            (self.df['Created'] >= recent_threshold) &
            (self.df['Tier'] == 3) &
            self.successful_inflow &
            (self.df['amount'] > min_amount) &
            (~self.df['Sender_Account'].isin(self.OUR_BANK_ACCOUNTS))
        ].copy()

        # Step 2: Loop over inflows and find any outflow within 24 hours
        for _, inflow in inflows.iterrows():
            inflow_idx = inflow.name
            inflow_time = inflow['Booking Date (Entry Date)']
            inflow_acc = inflow['Account ID (Transaction)']

            # Outflows within 24 hours
            outflows = self.df[
                (self.df['Account ID (Transaction)'] == inflow_acc) &
            (~self.df['Receiver_Account'].isin(self.OUR_BANK_ACCOUNTS)) &
                self.succesful_outflow_mask &
                (self.df['Booking Date (Entry Date)'] > inflow_time) &
                (self.df['Booking Date (Entry Date)'] <= inflow_time + pd.Timedelta(hours=24))
            ]

            if not outflows.empty and outflows['amount'].sum() >=similarity_score * inflow['amount']:
                # Flag inflow and outflow transactions
                flagged_txn_idx.add(inflow_idx)
                flagged_txn_idx.update(outflows.index)

        # Step 3: Apply flags in-place
        self.df.loc[
            self.df.index.isin(flagged_txn_idx),
            'flag_suspicious_rapid_cashout'
        ] = True

        # Step 4: Account-level summary
        summary = self.df.groupby('Account ID (Transaction)')[
            'flag_suspicious_rapid_cashout'
        ].any().reset_index()

        logging.info(
            f"Check completed. Flagged {summary['flag_suspicious_rapid_cashout'].sum()} "
            "accounts with suspicious rapid cashout"
        )

        return summary

    def flag_high_withdrawal_by_students(self,min_amount=1_000_000):
        """
        Flags high-value debit transactions by student accounts.
        """

        logging.info('Running high student withdrawal check...')

        # Ensure consistent case
        self.users_df['BusinessLine'] = self.users_df['BusinessLine'].str.strip().str.lower()

        # Identify student accounts
        student_accounts = self.users_df[self.users_df['BusinessLine'] == 'student']['Account ID (Transaction)'].unique()

        # Initialize in-place flag
        self.df['flag_high_student_withdrawal'] = False

        # Filter high-value debit transactions
        high_withdrawals = self.df[
            (self.df['Account ID (Transaction)'].isin(student_accounts)) &
            self.succesful_outflow_mask &
            (self.df['amount'] >= min_amount)
        ]

        # Flag the transactions IN PLACE
        self.df.loc[high_withdrawals.index, 'flag_high_student_withdrawal'] = True

        # Account-level summary
        summary = self.df.groupby('Account ID (Transaction)')['flag_high_student_withdrawal'].any().reset_index()

        logging.info(f"Check completed. Flagged {summary['flag_high_student_withdrawal'].sum()} student accounts with high withdrawals.")

        return summary

    def flag_multiple_wallets_to_same_receiver(self, min_wallets=5,min_amount=100_000):
        """
        Flags transactions where multiple wallets send to the same receiver account.
        - Flags all transactions sent to receivers with >= min_wallets unique senders.
        - Updates transactions_df in-place with transaction-level flags.
        - Returns account-level summary.
        """

        logging.info('Running multiple wallets to same receiver check...')

        # Initialize in-place flag column
        self.df['flag_shared_receiver'] = False

        # Filter eligible debit transactions
        df = self.df[
            (self.df['Receiver_Bank'] !=self.OUR_BANK_NAME) &
            (self.df['Receiver_Bank'].notna()) &
            (~self.df['Receiver_Account'].isin(self.OUR_BANK_ACCOUNTS)) &
            self.succesful_outflow_mask &
            (self.df['amount'] >=min_amount) 
        ].copy()

        # Count unique senders per receiver (vectorized)
        receiver_counts = df.groupby('Receiver_Account')['Account ID (Transaction)'].nunique()
        flagged_receivers = receiver_counts[receiver_counts >= min_wallets].index

        # Flag all transactions sent to flagged receivers
        flagged_txns = df[df['Receiver_Account'].isin(flagged_receivers)]
        self.df.loc[flagged_txns.index, 'flag_shared_receiver'] = True

        # Account-level summary
        summary = self.df.groupby('Account ID (Transaction)')['flag_shared_receiver'].any().reset_index()

        logging.info(f"Check completed. Flagged {summary['flag_shared_receiver'].sum()} accounts sending to shared receivers.")

        return summary

    def flag_loan_after_large_deposit(self, threshold=1_000_000, window_days=1):
        """
        Flags transactions where a customer receives a large deposit
        and applies for a loan within a short window.
        """

        logging.info('Running loan-after-large-deposit check...')

        # Ensure flag column exists
        self.df['flag_loan_after_deposit'] = False
        flagged_txn_idx = set()

        # Separate large deposits
        deposits = self.df[
            self.successful_inflow &
            (self.df['amount'] >= threshold)
        ].copy()

        # Filter loan transactions
        loans = self.df[
            self.loan_mask
        ].copy()

        # Merge deposits and loans per account to identify loans after deposits
        # Use cross join per customer for efficiency
        deposits_loans = deposits.merge(
            loans,
            on='Account ID (Transaction)',
            suffixes=('_deposit', '_loan')
        )

        # Check if loan occurred within window_days after deposit
        deposits_loans['within_window'] = (
            (deposits_loans['Booking Date (Entry Date)_loan'] > deposits_loans['Booking Date (Entry Date)_deposit']) &
            (deposits_loans['Booking Date (Entry Date)_loan'] <= deposits_loans['Booking Date (Entry Date)_deposit'] + pd.Timedelta(days=window_days))
        )

        # Flag transactions where condition is True
        flagged_pairs = deposits_loans[deposits_loans['within_window']]

        # Flag both the deposit and loan transactions
        flagged_txn_idx.update(flagged_pairs['Booking Date (Entry Date)_deposit'].index)
        flagged_txn_idx.update(flagged_pairs['Booking Date (Entry Date)_loan'].index)

        # Apply flags in-place
        self.df.loc[
            self.df.index.isin(flagged_txn_idx),
            'flag_loan_after_deposit'
        ] = True

        # Account-level summary
        summary = self.df.groupby('Account ID (Transaction)')['flag_loan_after_deposit'].any().reset_index()

        logging.info(f"Check completed. Flagged {summary['flag_loan_after_deposit'].sum()} accounts with loan after large deposit.")

        return summary

    def flag_structured_loan_withdrawals(self, window_days=7):
        """
        Flags structured loan withdrawals:
        - Loan disbursement accounts withdrawing to the same receiver within a short window
        - Flags transactions in-place and returns account-level summary
        """

        logging.info('Running structured loan withdrawals check...')

        # Initialize in-place flag column
        self.df['flag_structured_loan_withdrawal'] = False
        flagged_txn_idx = set()

        # Step 1: Identify loan disbursements
        loan_txns = self.df[self.loan_mask][
            ['Account ID (Transaction)', 'Booking Date (Entry Date)']
        ].rename(columns={'Booking Date (Entry Date)': 'loan_date'}).copy()

        # Step 2: Identify eligible debit withdrawals
        debit_txns = self.df[
            self.succesful_outflow_mask &
            (~self.df['Receiver_Bank'].isin([self.OUR_BANK_NAME])) &
            (~self.df['Receiver_Account'].isin(self.OUR_BANK_ACCOUNTS)) &
            (self.df['Receiver_Account'].notna())
        ].copy()

        # Step 3: Merge loan accounts with debit withdrawals
        merged = debit_txns.merge(
            loan_txns,
            on='Account ID (Transaction)',
            how='inner'
        )

        # Step 4: Keep only withdrawals within the window after loan date
        merged = merged[
            (merged['Booking Date (Entry Date)'] > merged['loan_date']) &
            (merged['Booking Date (Entry Date)'] <= merged['loan_date'] + pd.Timedelta(days=window_days))
        ]

        if merged.empty:
            logging.info("No structured loan withdrawals detected.")

            # Return account-level summary
            summary = self.df.groupby('Account ID (Transaction)')['flag_structured_loan_withdrawal'].any().reset_index()
            return summary

        # Step 5: Identify receivers with ≥ 2 distinct loan accounts
        receiver_groups = merged.groupby('Receiver_Account')['Account ID (Transaction)'].nunique()
        suspicious_receivers = set(receiver_groups[receiver_groups >= 2].index)

        # Step 6: Flag all transactions sent to suspicious receivers
        flagged_txns = merged[merged['Receiver_Account'].isin(suspicious_receivers)]
        flagged_txn_idx.update(flagged_txns.index)

        # Step 7: Apply flags in-place
        self.df.loc[
            self.df.index.isin(flagged_txn_idx),
            'flag_structured_loan_withdrawal'
        ] = True

        # Step 8: Account-level summary
        summary = self.df.groupby('Account ID (Transaction)')['flag_structured_loan_withdrawal'].any().reset_index()

        logging.info(f"Check completed. Flagged {summary['flag_structured_loan_withdrawal'].sum()} accounts with structured loan withdrawals.")

        return summary

############################################################################################################################################

    def run_all_fraud_checks(self):
        # Run all checks
        flags = [
            self.flag_negative_balance(),
            self.flag_round_figure(),
            self.tier1_daily_limit(),
            self.tier1_balance_limit(),
            self.flag_odd_hour(),
            self.active_dormant_check(),
            self.flag_high_transaction_velocity(),
            self.flag_structured_outflow(),
            self.flag_structured_transfers(),
            self.flag_structured_large_inflow(),
            self.flag_early_loan_repayment_per_disbursement(),
            self.flag_linked_accounts_by_bvn(),
            self.flag_transaction_surge(),
            self.single_trans_check(),
            self.flag_large_inflows_on_new_accounts(),
            self.flag_outflow_on_new_accounts(),
            self.flag_rapid_withdrawals_by_customer(),
            self.flag_duplicate_outflow(),
            self.flag_third_party_repayments(),
            self.flag_linked_transfers(),
            self.flag_suspicious_rapid_cashout(),
            self.flag_high_withdrawal_by_students(),
            self.flag_multiple_wallets_to_same_receiver(),
            self.flag_loan_after_large_deposit(),
            self.flag_structured_loan_withdrawals()
        ]
        logging.info("Checks completed.")
        merged_flags = self.users_df.copy()
        
        
        # Merge all flags into one Table
        for flag_df in flags:
            if not flag_df.empty:
                
                merged_flags = merged_flags.merge(flag_df, on='Account ID (Transaction)', how='left')

        # Replace NaN flags with False
        flag_cols = [col for col in merged_flags.columns if "flag" in col]
        txn_flag_cols = [col for col in self.df.columns if "flag" in col]

        # Replace boolean with 1 and 0
        merged_flags[flag_cols] = (merged_flags[flag_cols]
                                    .fillna(False)
                                    .astype("int8")
                                )
        summary = summary = (
        merged_flags[flag_cols]
        .sum()
        .rename('Total Users Flagged')
        .reset_index()
        .rename(columns={'index': 'Flag Name'}))

        txn_counts = self.df[txn_flag_cols].sum()

        txn_values = (
            self.df[txn_flag_cols]
            .multiply(self.df['amount'], axis=0)
            .sum()
        )

        summary_txn = (
            pd.concat([txn_counts, txn_values], axis=1)
            .reset_index()
            .rename(columns={
                'index': 'Flag Name',
                0: 'Txn Count',
                1: 'Txn Value'
            })
        )

        final_summary = summary.merge(summary_txn, on= 'Flag Name', how='left')

        return merged_flags, final_summary
