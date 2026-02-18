from pathlib import Path
import pandas as pd
from utility.data_processing import clean_trans_data

def load_data(file_path):

    excel_files = list(Path(file_path).rglob('*.xls*'))
    
    if not excel_files: 
        raise FileNotFoundError(f"No Excel files found in the '{file_path}' directory.")
    
    # Load all Excel files into a list of DataFrames 
    dataframes = [pd.read_excel(file) for file in excel_files] 
    
    # Concatenate all DataFrames into one 
    merged_df = pd.concat(dataframes, ignore_index=True)

    return merged_df

def data_loader(trans_df_path,signups_path, user_data_path,flag_path,loan_path):

    # Loan data
    flag_data = load_data(flag_path)
    loan_data = load_data(loan_path)[['category','MambuTransactionId']].rename(columns={'category':'type_','MambuTransactionId':"Transaction ID"})
    trans_df = load_data(trans_df_path).merge(loan_data, on="Transaction ID", how='left')
    user_other_data = load_data(user_data_path)[['MambuAccountID','Tier','BusinessLine','BVN']]
    user_data =  load_data(signups_path).merge(user_other_data,left_on='Account ID',right_on='MambuAccountID',how='left')
    
    # Drop redundant columns
    user_data.drop(['Account','Balance','Last Modified','Account State',
                    'MambuAccountID','Account Officer'],axis=1, inplace=True)
    
    user_data['Created'] = pd.to_datetime(user_data['Created'])
    user_data = user_data.rename(columns={'Account ID':'Account ID (Transaction)'})

    trans_data = clean_trans_data(trans_df)
    trans_data = trans_data.merge(user_data[['Created','Tier','Account ID (Transaction)']],
                                  on='Account ID (Transaction)',how='left')
 
    return flag_data, trans_data, user_data