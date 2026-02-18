import pandas as pd
import re
import numpy as np

def clean_trans_data(trans_df):

    notes = trans_df['Notes (Transaction)'].fillna('')
    txn_type = trans_df['Type (Transaction)'].fillna('').str.strip().str.casefold()
    channel = trans_df['Channel (Transaction)'].fillna('').str.strip().str.casefold()

    noise_mask = (
        notes.str.contains(
            r'Fee\\|VAT\\|Small Balance|AMF \+|bank ch|BranchSweep|CHARGES ON CLIENT WITHDRAWAL',
            case=False,
            regex=True
        )
        | (txn_type == 'branch changed')
        | (channel == 'stamp duties')
        | (channel == 'pos transaction receivable')
        
    )

    # Remove Fees, Charges from the transaction data
    df = trans_df.loc[~noise_mask].copy()

    # Identify reversal types
    is_adjustment = df['Type (Transaction)'].str.contains('adjustment', case=False, na=False)
    has_rvsl_id = df['Notes (Transaction)'].str.contains(r'RVSL_|RVSLBP_', case=False, na=False)
    has_rvsl_word = df['Notes (Transaction)'].str.contains(
        r'RVSL\\|RVSL-|RVSL -|Revers', case=False, na=False
    )

    # Filter out reversal of type 1 - used for adjusting transaction
    reversal_data1 = df[is_adjustment]

    # Filter out reversal of type 2 - system generated reversal
    reversal_data2 = df[~is_adjustment & has_rvsl_id]

    # Filter out reversal of type 3 - inconsistent
    reversal_data3 = df[~is_adjustment & ~has_rvsl_id & has_rvsl_word]

    # Remove all of these reversals from the transaction data
    cleaned_data = df.loc[~(is_adjustment | notes.str.contains('rvsl|revers', case=False))].copy()

    # Build match keys (vectorized)
    def build_key(x):
        return (
            x['Booking Date (Entry Date)'].astype(str) + '_' +
            x['Amount (Transaction)'].abs().astype(str) + '_' +
            x['Account ID (Transaction)'].astype(str)
        )
    
    # Then, create a unique key for these dataframes to match on
    cleaned_data['match_key'] = build_key(cleaned_data)
    reversal_data1 = reversal_data1.copy()
    reversal_data2 = reversal_data2.copy()

    reversal_data1['match_key'] = build_key(reversal_data1)
    reversal_data2['match_key'] = build_key(reversal_data2)

    # remove reversal type 1 if already exist in type 2 - to prevent duplication
    reversal_data = reversal_data2.loc[~reversal_data2['match_key'].isin(reversal_data1['match_key'])]
    
    # obtain reversal identifiers from the reversal type 3
    reversal_ids = set(reversal_data3['Notes (Transaction)']
                       .apply(check_rvsl)
                       .dropna()
                       .astype(str)
                       )

    # Obtain status based on if transaction was reversed or successful
    cleaned_data['status'] = 'successful'
    cleaned_data.loc[cleaned_data['match_key'].isin(reversal_data1['match_key']),'status'] = 'reversed'
    cleaned_data.loc[cleaned_data['Notes (Transaction)'].apply(lambda note: any(rid in str(note) for rid in reversal_ids)),'status'] = 'reversed'
    cleaned_data.loc[cleaned_data['Transaction ID'].isin(reversal_data['Notes (Transaction)'].apply(extract_rvsl_id).astype(int)),'status'] = 'reversed'
    
    # Categorization 
    cleaned_data['category'] = np.where(
        cleaned_data['type_'].notna(),
        cleaned_data['type_'],
        cleaned_data.apply(
            lambda r: trans_category(r['Notes (Transaction)'], r['Amount (Transaction)']),
            axis=1
        )
    )
    # Clean any literal backslashes (if mistakenly included in the Narration)
    cleaned_data['Notes (Transaction)'] = cleaned_data['Notes (Transaction)'].str.replace(r'\\', '', regex=True)
    
    # Regex pattern using pipe '|' as delimiter
    pattern = (
        r'(?P<Receiver_Account>[^|]*)\|'
        r'(?P<Receiver_Name>[^|]*)\|'
        r'(?P<Receiver_Bank>[^|]*)\|'
        r'(?P<Sender_Account>[^|]*)\|'
        r'(?P<Sender_Name>[^|]*)\|'
        r'(?P<Sender_Bank>[^|]*)\|'
        r'(?P<Amount>[^|]*)\|'
        r'(?P<Fee>[^|]*)\|'
        r'(?P<Transaction_ID>[^|]*)\|'
        r'(?P<Transaction_Ref>[^|]*)\|'
        r'(?P<Transaction_DateTime>[^|]*)\|'
        r'(?P<Status>[^|]*)\|'
        r'(?P<Transaction_Type>[^|]*)\|'
        r'(?P<Channel>[^|]*)'
    )
    
    # Apply regex to extract fields
    extracted = cleaned_data['Notes (Transaction)'].str.extract(pattern)
    
    # Join back to the original DataFrame
    cleaned_data = cleaned_data.join(extracted)

    # derive additional columns
    cleaned_data['ledger_type'] = np.where(cleaned_data['Amount (Transaction)']<0, 'debit','credit')

    cleaned_data['amount'] = cleaned_data['Amount (Transaction)'].abs().astype(float)
    cleaned_data['Booking Date (Entry Date)'] = pd.to_datetime(cleaned_data['Booking Date (Entry Date)'])
    cleaned_data['txn_date'] = cleaned_data['Booking Date (Entry Date)'].dt.date
    cleaned_data['txn_hour'] = cleaned_data['Booking Date (Entry Date)'].dt.hour
    
    # Remove redundant columns
    final_data = cleaned_data.drop(['Entry ID','type_','Type (Transaction)','Amount (Transaction)', 'match_key','Fee', 
                                'Transaction_ID','Status','Internal Transfer (Transaction)','Transaction_Ref',
                                'Channel','Transaction_Type','Amount'],axis=1)
    
    reversal_data = reversal_data.copy()
    
    reversal_data['category'] = reversal_data.apply(
        lambda row: trans_category(row['Notes (Transaction)'], -1*row['Amount (Transaction)']),
        axis=1
    )
    return final_data

def extract_rvsl_id(note):
    
    if pd.isna(note):
        return None

    if "RVSL_" in note:
        match = re.search(r"RVSL_[A-Z]+_(\d+)", note)
        if match:
            return match.group(1)

    elif "RVSLBP_" in note:
        match = re.search(r"RVSLBP_[A-Z]+_(\d+)", note)
        if match:
            return match.group(1)
    else:
        return None


def check_rvsl(note):
    
    if pd.isna(note):
        return None

    if "RVSL\\" in note:
        try:
            # Get text after 'RVSL_'
            part = note.split("REP-")[1]
            # Get text before first '|'
            part = part.split(r"\|")[0]
            return part
        except IndexError:
            return None

    elif "RVSL -" in note or "RVSL-" in note:
        try:
            part = note.split("ETRN_")[1]
            part = part.split(r"_")[0]
            return part
        except IndexError:
            return None
    elif "Reversal: " in note:
        try:
            part = note.split("Reversal: ")[1]
            part = part.split("\n")[0]
            return part
        except IndexError:
            return None
    elif "Reversal " in note:
        try:
            part = note.split("Reversal ")[1]
            part = part.split("\t")[0]
            return part
        except IndexError:
            return None
            
    elif "reverse" in note:
            return "REP-2025031115105951609"
    else:
        return None

def trans_category(note, amount):

    note = str(note) if pd.notna(note) else ""
    
    if "LOANDIS" in note and amount > 0:
        return "Nano"

    if "PSCRDBP" in note and amount > 0:
        return "POS"

    if "Repayment" in note and amount < 0 and "Repayment for  DIVINE CHIDIMA AGOM" not in note:
        return "Loan Repayment"

    if "ATPRCH" in note and amount < 0:
        return "Airtime"
        
    if "DTPRCH" in note and amount < 0:
        return "Data"
        
    if "CTVPRCH" in note and amount < 0:
        return "Cable TV"
        
    if "ETPRCH" in note and amount < 0:
        return "Electricity"
        
    if amount > 0:
        return "Inflow"

    if amount < 0:
        return "Outflow"