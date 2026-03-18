import os
import json
from django.conf import settings
from django.db import transaction

MODEL_DIR    = os.path.join(settings.BASE_DIR, 'analytics', 'ml')
METRICS_FILE = os.path.join(MODEL_DIR, 'model_metrics.json')

Z_95             = 1.96
FALLBACK_CI_PCT  = 0.15


def _load_metrics():
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE) as fh:
            return json.load(fh)
    return {}


def run_predictions(metrics=None):
    """
    Load trained models, generate a rolling 24-month forecast per product,
    compute model-specific 95% CI from residual std, and bulk-insert into
    Sales_Forecast.

    Heavy imports are deferred so Django startup never touches pandas/sklearn.
    """
    # ── Lazy imports ──────────────────────────────────────────────────────────
    import pandas as pd
    import numpy as np
    import joblib
    from datetime import date
    from dateutil.relativedelta import relativedelta
    from analytics.models import Product, Sales_Forecast, Sales_Transaction
    from analytics.ml.features import fetch_and_prepare_data

    if metrics is None:
        metrics = _load_metrics()

    models      = {}
    model_names = ['XGBoost', 'Prophet', 'ARIMA', 'LSTM']

    for m_name in model_names:
        m_path = os.path.join(MODEL_DIR, f"{m_name.lower()}_model.pkl")
        if os.path.exists(m_path):
            try:
                models[m_name] = joblib.load(m_path)
            except Exception as exc:
                print(f"Warning: could not load {m_name} — {exc}")
        else:
            print(f"Warning: {m_name} model not found at {m_path}.")

    if not models:
        print("No models found. Cannot run predictions.")
        return

    print(f"Loaded {len(models)} models: {list(models.keys())}")

    df = fetch_and_prepare_data()
    if df.empty:
        print("No historical data available.")
        return

    products = Product.objects.all()
    if not products.exists():
        print("No products in database.")
        return

    try:
        max_date   = Sales_Transaction.objects.latest('transaction_date').transaction_date
        start_date = max_date.replace(day=1) + relativedelta(months=1)
    except Sales_Transaction.DoesNotExist:
        start_date = date.today().replace(day=1)

    HORIZON_MONTHS = 24

    Sales_Forecast.objects.all().delete()
    print("Cleared existing forecasts.")
    print(f"Generating forecasts for {products.count()} products × "
          f"{HORIZON_MONTHS} months from {start_date}...")

    new_forecasts = []

    for model_name, model in models.items():
        model_metrics = metrics.get(model_name, {})
        mape    = model_metrics.get('mape', 5.0)
        res_std = model_metrics.get('residual_std', None)

        for product in products:
            prod_id    = product.product_id
            base_price = float(product.base_price)

            prod_history = df[df['product_id'] == prod_id]

            if not prod_history.empty:
                lag_1 = float(prod_history.iloc[-1]['total_revenue'])
                lag_2 = float(prod_history.iloc[-2]['total_revenue']) if len(prod_history) > 1 else 0.0
            else:
                lag_1 = 0.0
                lag_2 = 0.0

            current_date = start_date

            for _ in range(HORIZON_MONTHS):
                features = pd.DataFrame([{
                    'product_id':    prod_id,
                    'year':          current_date.year,
                    'month_num':     current_date.month,
                    'base_price':    base_price,
                    'lag_1_revenue': lag_1,
                    'lag_2_revenue': lag_2,
                }])

                pred_revenue = max(0.0, float(model.predict(features)[0]))

                if res_std is not None and res_std > 0:
                    half_width = Z_95 * res_std
                else:
                    half_width = pred_revenue * FALLBACK_CI_PCT

                lower_bound = max(0.0, pred_revenue - half_width)
                upper_bound = pred_revenue + half_width

                new_forecasts.append(Sales_Forecast(
                    product=product,
                    forecast_date=current_date,
                    forecast_revenue=round(pred_revenue, 2),
                    lower_bound=round(lower_bound, 2),
                    upper_bound=round(upper_bound, 2),
                    model_version=model_name,
                    mape=round(mape, 2),
                ))

                lag_2 = lag_1
                lag_1 = pred_revenue
                current_date += relativedelta(months=1)

    with transaction.atomic():
        Sales_Forecast.objects.bulk_create(new_forecasts, batch_size=500)

    print(f"Successfully saved {len(new_forecasts)} forecasts to the database.")
