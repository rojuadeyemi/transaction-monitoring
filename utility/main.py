from utility.rule_engine import RuleEngine
from utility.auxiliary_functions import data_loader
from utility.risk_scoring import RiskScoringEngine
import os
from datetime import datetime, timedelta
import pandas as pd
import logging
import time

# Configure logging once
logging.basicConfig(
    filename='flags.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def screen_transaction(save_path, trans_df_path,signups_path, 
                       user_data_path,flag_path,loan_path, 
                       startdate,enddate):
    start_time = time.time()
    logging.info("Transaction screening initiated...")

    # Loan data
    flag_data, trans_df, users_df = data_loader(trans_df_path,signups_path, 
                                                user_data_path,flag_path,
                                                loan_path)
    logging.info("Data loading completed.")

    # Extract only current period data
    trans_df = trans_df[trans_df['Booking Date (Entry Date)'] <= enddate]
    df = trans_df[(trans_df['Booking Date (Entry Date)'] >= startdate)].copy()
    
    logging.info("Current period data obtained ")

    # Obtain Results
    rule = RuleEngine(df, trans_df, users_df)
    result, summary = rule.run_all_fraud_checks()
    summary_report = flag_data.merge(summary,on='Flag Name',how='left')

    # Add Risk Score
    engine = RiskScoringEngine(flag_data)
    df = engine.compute_score(df)
    result = engine.compute_score(result)

    logging.info("Saving the results...")

    #Save necessary data for further analysis
    os.makedirs(save_path,exist_ok=True)
    df.drop(['txn_date','txn_hour','Created','Tier'],axis=1, inplace=True)

    # Create an ExcelWriter object
    timestamp = datetime.today().strftime("%Y-%m-%d")
    file_path = os.path.join(save_path,f"Transaction_screening_report_{timestamp}.xlsx")
    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        summary_report.to_excel(writer, sheet_name='Summary', index=False)
        result.to_excel(writer, sheet_name='Risk Report', index=False)
        df.to_excel(writer, sheet_name='Transactions', index=False)

    logging.info("Results saved successfully")
    
    # estimate runtime
    end_time = time.time()
    runtime = timedelta(seconds=end_time - start_time)

    logging.info(f"Transaction screening completed.\nTotal runtime: {runtime} (HH:MM:SS)\n\n")

if __name__ == '__main__':

    # Provide data paths
    trans_df_path = r'url_to_payment_data'
    signups_path = r'url_to_user_data'
    user_data_path = r"url_to_account_info"
    flag_path = r"url_to_rules_and_weight"
    loan_path = r"url_to_loan_data" # This helps to identify loan transactions from the payment data

    # Enter where to store the result data
    save_path = r'url_to_onedrive_storage'


    # Shedule - Define date range for the transaction
    enddate = datetime.today() - timedelta(days=1)
    startdate = datetime.today() - timedelta(days=8)

    # On-Demand - Manually entered dates as strings
    #startdate = pd.to_datetime("2025-10-01")
    #enddate = pd.to_datetime("2025-12-31")
    
    #Run the flags
    screen_transaction(save_path, trans_df_path,signups_path, user_data_path,flag_path, loan_path, startdate,enddate)
