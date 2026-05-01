import pandas as pd
import numpy as np
import holidays

def build_features(proc_dir='dataset/processed/'):
    print("Xây dựng đặc trưng dựa trên phân tích Seasonality & Trend...")
    df = pd.read_csv(f"{proc_dir}daily_sales.csv", parse_dates=['Date'])

    df['month'] = df['Date'].dt.month
    df['day_of_week'] = df['Date'].dt.dayofweek

    df['is_peak_season'] = df['month'].isin([4, 5, 6]).astype(int)

    df['rev_lag_364'] = df['rev_log'].shift(364)
    df['cogs_lag_364'] = df['cogs_log'].shift(364)

    df['rev_lag_7'] = df['rev_log'].shift(7)

    day_of_year = df['Date'].dt.dayofyear
    df['sin_year'] = np.sin(2 * np.pi * day_of_year / 365.25)
    df['cos_year'] = np.cos(2 * np.pi * day_of_year / 365.25)

    df['rev_roll_mean_7'] = df['rev_log'].shift(7).rolling(window=7).mean()
    month_map = df.groupby('month')['rev_log'].mean().to_dict()
    df['month_avg_rev'] = df['month'].map(month_map)

    day_map = df.groupby('day_of_week')['rev_log'].mean().to_dict()
    df['day_avg_rev'] = df['day_of_week'].map(day_map)

    df['rev_lag_14'] = df['rev_log'].shift(14)
    df['rev_lag_28'] = df['rev_log'].shift(28)

    df['day_of_month'] = df['Date'].dt.day
    df['is_payday_period'] = df['day_of_month'].isin([1,2,15, 30]).astype(int)
    df['rev_cogs_ratio_7d'] = df['rev_log'].shift(7) / (df['cogs_log'].shift(7) + 1e-9)
    df['is_high_spending_period'] = df['day_of_month'].isin([2, 3, 4, 16, 18]).astype(int)

    vn_holidays = holidays.VN()
    df['is_holiday'] = df['Date'].apply(lambda x: 1 if x in vn_holidays else 0)
    
    df['holiday_nearby'] = df['is_holiday'].shift(-1).fillna(0)
    
    df = df.dropna().reset_index(drop=True)
    
    df.to_csv(f"{proc_dir}final_features_matrix.csv", index=False)
    print(f" Đã tạo {df.shape[1]} đặc trưng. Sẵn sàng cho huấn luyện.")
    return df

if __name__ == '__main__':
    build_features()