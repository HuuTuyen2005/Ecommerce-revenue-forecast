import pandas as pd
import numpy as np
import os

def preprocess_data(raw_dir='dataset/raw/', proc_dir='dataset/processed/'):
    if not os.path.exists(proc_dir):
        os.makedirs(proc_dir)

    print(" Preprocessing Sales Data with Log Transformation...")
    sales = pd.read_csv(f"{raw_dir}sales.csv", parse_dates=['Date'])
    
    sales = sales.sort_values('Date').reset_index(drop=True)

    sales['rev_log'] = np.log1p(sales['Revenue'])
    sales['cogs_log'] = np.log1p(sales['COGS'])

    sales.to_csv(f"{proc_dir}daily_sales.csv", index=False)
    print(" Preprocessing hoàn tất.")

if __name__ == "__main__":
    preprocess_data()


