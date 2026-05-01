import pandas as pd
import numpy as np
import joblib
import os
import holidays
from datetime import timedelta

def make_prediction(proc_dir='dataset/processed/', model_dir='models/', output_file='submission.csv'):
    print(" Khởi động quy trình dự báo Revenue (Recursive) với COGS cố định...")

    try:
        model_xgb_rev = joblib.load(f"{model_dir}xgb_rev.pkl")
        model_cb_rev = joblib.load(f"{model_dir}cb_rev.pkl")
        feature_names = joblib.load(f"{model_dir}feature_names.pkl")
        print(" Đã load thành công các model Revenue.")
    except Exception as e:
        print(f" Lỗi load model: {e}")
        return

    try:
        sample_sub = pd.read_csv('dataset/raw/sample_submission.csv')
        sample_sub['Date'] = pd.to_datetime(sample_sub['Date'])
        cogs_lookup = sample_sub.set_index('Date')['COGS'].to_dict()
        predict_dates = sample_sub['Date'].unique()
    except Exception as e:
        print(f" Lỗi load file sample_submission.csv: {e}")
        return

    df_history = pd.read_csv(f"{proc_dir}final_features_matrix.csv", parse_dates=['Date'])
    full_df = df_history.copy()

    df_history['month'] = df_history['Date'].dt.month
    df_history['day_of_week'] = df_history['Date'].dt.dayofweek
    month_map = df_history.groupby('month')['rev_log'].mean().to_dict()
    day_map = df_history.groupby('day_of_week')['rev_log'].mean().to_dict()
    vn_holidays = holidays.VN()

    print(f" Đang dự báo cho {len(predict_dates)} ngày...")

    for current_date in predict_dates:
        current_cogs = cogs_lookup.get(current_date, 0)
        current_cogs_log = np.log1p(current_cogs)

        new_row = pd.DataFrame({'Date': [current_date]})
        new_row['month'] = new_row['Date'].dt.month
        new_row['day_of_week'] = new_row['Date'].dt.dayofweek
        new_row['day_of_month'] = new_row['Date'].dt.day
        new_row['is_peak_season'] = new_row['month'].isin([4, 5, 6]).astype(int)

        def get_lag(date, lag_days, col):
            target_date = date - timedelta(days=lag_days)
            val = full_df.loc[full_df['Date'] == target_date, col]
            return val.iloc[0] if not val.empty else 0

        new_row['rev_lag_364'] = get_lag(current_date, 364, 'rev_log')
        new_row['cogs_lag_364'] = get_lag(current_date, 364, 'cogs_log')
        new_row['rev_lag_7'] = get_lag(current_date, 7, 'rev_log')
        new_row['rev_lag_14'] = get_lag(current_date, 14, 'rev_log')
        new_row['rev_lag_28'] = get_lag(current_date, 28, 'rev_log')
        
        lags_7d = [get_lag(current_date, i, 'rev_log') for i in range(7, 14)]
        new_row['rev_roll_mean_7'] = np.mean(lags_7d)
        
        day_of_year = current_date.timetuple().tm_yday
        new_row['sin_year'] = np.sin(2 * np.pi * day_of_year / 365.25)
        new_row['cos_year'] = np.cos(2 * np.pi * day_of_year / 365.25)
        
        new_row['is_holiday'] = 1 if current_date in vn_holidays else 0
        new_row['holiday_nearby'] = 1 if (current_date + timedelta(days=1)) in vn_holidays else 0
        new_row['is_payday_period'] = new_row['day_of_month'].isin([1, 2, 15, 30]).astype(int)
        new_row['is_high_spending_period'] = new_row['day_of_month'].isin([2, 3, 4, 16, 18]).astype(int)
        
        rev_7 = get_lag(current_date, 7, 'rev_log')
        cogs_7 = get_lag(current_date, 7, 'cogs_log')
        new_row['rev_cogs_ratio_7d'] = rev_7 / (cogs_7 + 1e-9)
        
        new_row['month_avg_rev'] = month_map.get(new_row['month'].iloc[0], np.mean(list(month_map.values())))
        new_row['day_avg_rev'] = day_map.get(new_row['day_of_week'].iloc[0], np.mean(list(day_map.values())))

        X = new_row[feature_names]
        p_xgb_rev = model_xgb_rev.predict(X)[0]
        p_cb_rev = model_cb_rev.predict(X)[0]
        rev_log_pred = (p_xgb_rev * 0.9) + (p_cb_rev * 0.1)
        
        new_row['rev_log'] = rev_log_pred
        new_row['cogs_log'] = current_cogs_log  
        
        full_df = pd.concat([full_df, new_row], ignore_index=True)

    submission_df = full_df[full_df['Date'].isin(predict_dates)].copy()
    
    submission_df['Revenue'] = np.expm1(submission_df['rev_log']).clip(lower=0).round(2)
    
    final_output = sample_sub[['Date', 'COGS']].merge(
        submission_df[['Date', 'Revenue']], 
        on='Date', 
        how='left'
    )

    final_output = final_output[['Date', 'Revenue', 'COGS']]
    final_output['Date'] = final_output['Date'].dt.strftime('%Y-%m-%d')
    
    final_output.to_csv(output_file, index=False)
    print(f" Đã tạo file thành công: {output_file}")
    print(final_output.head())

if __name__ == '__main__':
    make_prediction()