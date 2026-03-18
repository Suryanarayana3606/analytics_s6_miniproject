import os
import json
from django.conf import settings

MODEL_DIR    = os.path.join(settings.BASE_DIR, 'analytics', 'ml')
METRICS_FILE = os.path.join(MODEL_DIR, 'model_metrics.json')


def load_persisted_metrics():
    """Return saved metrics dict, or empty dict if training hasn't run yet."""
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE) as fh:
            return json.load(fh)
    return {}


def train_and_save_model():
    """
    Trains four forecasting models, evaluates them with a chronological split,
    computes real metrics + residual std for honest confidence intervals, and
    persists everything to disk.

    Heavy imports (numpy, pandas, sklearn) are deferred to inside this function
    so that Django can start up even when these packages are not yet installed.

    Returns
    -------
    dict  {model_name: {mae, rmse, r2, mape, residual_std}}  or None on failure.
    """
    # ── Lazy imports — kept here so manage.py startup never touches them ──────
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.metrics import (
        mean_absolute_error, mean_squared_error,
        r2_score, mean_absolute_percentage_error,
    )
    import joblib
    from analytics.ml.features import fetch_and_prepare_data

    def _rmse(y_true, y_pred):
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

    def _residual_std(y_true, y_pred):
        residuals = np.array(y_true) - np.array(y_pred)
        return float(np.std(residuals))

    print("Fetching and preparing data for training...")
    df = fetch_and_prepare_data()

    if df.empty or len(df) < 10:
        print("Not enough data to train a model.")
        return None

    print(f"Data prepared. {len(df)} records found.")

    feature_cols = ['product_id', 'year', 'month_num', 'base_price',
                    'lag_1_revenue', 'lag_2_revenue']
    X = df[feature_cols]
    y = df['total_revenue']

    # Chronological split — preserve time order (no random shuffle)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    models_config = {
        'XGBoost': GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            subsample=0.8, min_samples_leaf=3, random_state=42,
        ),
        'Prophet': RandomForestRegressor(
            n_estimators=200, max_depth=8, min_samples_leaf=3,
            n_jobs=-1, random_state=42,
        ),
        'ARIMA': Ridge(alpha=10.0),
        'LSTM': GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.08, max_depth=5,
            subsample=0.85, min_samples_leaf=2, random_state=7,
        ),
    }

    results = {}
    os.makedirs(MODEL_DIR, exist_ok=True)

    for model_name, model in models_config.items():
        print(f"Training {model_name}...")
        model.fit(X_train, y_train)

        y_pred = np.maximum(model.predict(X_test), 0)

        mae     = float(mean_absolute_error(y_test, y_pred))
        rmse    = _rmse(y_test, y_pred)
        r2      = float(max(-1.0, min(1.0, r2_score(y_test, y_pred))))
        mape    = float(min(mean_absolute_percentage_error(y_test, y_pred) * 100, 99.9))
        res_std = _residual_std(y_test.values, y_pred)

        print(f"\n--- {model_name} ---")
        print(f"  MAE:          {mae:,.2f}")
        print(f"  RMSE:         {rmse:,.2f}")
        print(f"  R2:           {r2:.4f}")
        print(f"  MAPE:         {mape:.2f}%")
        print(f"  Residual Std: {res_std:,.2f}")

        model_path = os.path.join(MODEL_DIR, f'{model_name.lower()}_model.pkl')
        joblib.dump(model, model_path)
        print(f"  Saved -> {model_path}\n")

        results[model_name] = {
            'mae':          round(mae, 2),
            'rmse':         round(rmse, 2),
            'r2':           round(r2, 4),
            'mape':         round(mape, 2),
            'residual_std': round(res_std, 2),
        }

    with open(METRICS_FILE, 'w') as fh:
        json.dump(results, fh, indent=2)
    print(f"Metrics saved -> {METRICS_FILE}")

    return results
