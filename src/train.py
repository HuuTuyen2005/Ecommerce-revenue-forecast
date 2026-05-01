import pandas as pd
import numpy as np
import xgboost as xgb
import catboost as cb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

def train_model(proc_dir='dataset/processed/', model_dir='models/'):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    print(" Đang tải dữ liệu ma trận đặc trưng...")
    df = pd.read_csv(f"{proc_dir}final_features_matrix.csv", parse_dates=['Date'])
    
    train_df = df[df['Date'].dt.year < 2022].copy()
    val_df = df[df['Date'].dt.year == 2022].copy()

    drop_cols = ['Date', 'Revenue', 'COGS', 'rev_log', 'cogs_log']
    features = [c for c in df.columns if c not in drop_cols]
    
    X_train = train_df[features]
    y_train_rev = train_df['rev_log']
    
    X_val = val_df[features]
    y_val_rev = val_df['rev_log']

    print(f" Số lượng features sử dụng: {len(features)}")
    print(f" Train size: {X_train.shape[0]} | Val size: {X_val.shape[0]}")

    xgb_params = {
        'n_estimators': 2000,
        'max_depth': 6,
        'learning_rate': 0.01,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'n_jobs': -1,
        'objective': 'reg:absoluteerror', 
        'tree_method': 'hist',
        'early_stopping_rounds': 100,
        'random_state': 42
    }

    cb_params = {
        'iterations': 2000,
        'learning_rate': 0.01,
        'depth': 6,
        'loss_function': 'MAE',
        'eval_metric': 'MAE',
        'random_seed': 42,
        'early_stopping_rounds': 100,
        'verbose': 200
    }

    print("\nĐang huấn luyện hệ thống Ensemble cho Revenue...")

    model_xgb_rev = xgb.XGBRegressor(**xgb_params)
    model_xgb_rev.fit(X_train, y_train_rev, eval_set=[(X_val, y_val_rev)], verbose=False)
    
    model_cb_rev = cb.CatBoostRegressor(**cb_params)
    model_cb_rev.fit(X_train, y_train_rev, eval_set=(X_val, y_val_rev))

    preds_xgb_log = model_xgb_rev.predict(X_val)
    preds_cb_log = model_cb_rev.predict(X_val)
    
    final_preds_log = (preds_xgb_log * 0.9) + (preds_cb_log * 0.1)
    
    preds_real = np.expm1(final_preds_log)
    actual_real = np.expm1(y_val_rev)
    
    preds_real = np.maximum(0, preds_real)
    
    mae = mean_absolute_error(actual_real, preds_real)
    rmse = np.sqrt(mean_squared_error(actual_real, preds_real))
    r2 = r2_score(actual_real, preds_real)

    print("\n" + "="*45)
    print("  KẾT QUẢ ĐÁNH GIÁ REVENUE ENSEMBLE")
    print("="*45)
    print(f"  MAE  (Mean Absolute Error): {mae:,.2f} VND")
    print(f"  RMSE (Root Mean Squared Error): {rmse:,.2f} VND")
    print(f"  R² Score: {r2:.4f}")
    print("="*45)

    error_df = val_df[['Date', 'Revenue']].copy()
    error_df['Predicted'] = preds_real
    error_df['Abs_Error'] = np.abs(error_df['Revenue'] - error_df['Predicted'])
    
    print("\n Top 10 ngày có sai số lớn nhất (Outliers):")
    print(error_df.sort_values(by='Abs_Error', ascending=False).head(10))

    joblib.dump(model_xgb_rev, f"{model_dir}xgb_rev.pkl")
    joblib.dump(model_cb_rev, f"{model_dir}cb_rev.pkl")
    joblib.dump(features, f"{model_dir}feature_names.pkl")
    
    print("\n Đã lưu hệ thống mô hình Ensemble (Revenue Only).")

    print("\n Đang phân tích Feature Importance tổng hợp (90% XGB + 10% CB)...")
    
    xgb_imp = model_xgb_rev.feature_importances_
    
    cb_imp_raw = model_cb_rev.get_feature_importance()
    cb_imp = cb_imp_raw / np.sum(cb_imp_raw) 
    
    ensemble_importance = (xgb_imp * 0.9) + (cb_imp * 0.1)
    
    imp_df = pd.DataFrame({
        'Feature': features,
        'Importance': ensemble_importance
    }).sort_values(by='Importance', ascending=False)

    print("\n TOP 10 ĐẶC TRƯNG QUAN TRỌNG NHẤT:")
    print("-" * 45)
    print(imp_df.head(10).to_string(index=False))
    print("-" * 45)

if __name__ == "__main__":
    train_model()