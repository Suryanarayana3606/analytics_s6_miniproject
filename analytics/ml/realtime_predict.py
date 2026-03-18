"""
Real-time prediction engine.

Loads trained model .pkl files once per process (module-level cache),
then runs model.predict() on demand for any feature combination.
No DB writes — results are returned directly to the caller.
"""
import os
import json
from django.conf import settings

MODEL_DIR    = os.path.join(settings.BASE_DIR, 'analytics', 'ml')
METRICS_FILE = os.path.join(MODEL_DIR, 'model_metrics.json')

Z_95 = 1.96

# _MODEL_CACHE / _METRICS_CACHE: loaded once per process, invalidated after retraining
_MODEL_CACHE: dict   = {}
_METRICS_CACHE: dict = {}


def _load_model(model_name: str):
    if model_name not in _MODEL_CACHE:
        try:
            import joblib
            path = os.path.join(MODEL_DIR, f"{model_name.lower()}_model.pkl")
            if not os.path.exists(path):
                return None
            _MODEL_CACHE[model_name] = joblib.load(path)
        except Exception as exc:
            print(f"[realtime_predict] Failed to load {model_name}: {exc}")
            return None
    return _MODEL_CACHE[model_name]


def _load_metrics() -> dict:
    if not _METRICS_CACHE:
        if os.path.exists(METRICS_FILE):
            with open(METRICS_FILE) as fh:
                _METRICS_CACHE.update(json.load(fh))
    return _METRICS_CACHE


def _ci_half_width(predicted: float, mape: float, res_std: float) -> float:
    """
    Choose the most appropriate CI half-width for a single-product prediction.

    The model was trained on data aggregated across ALL products, so res_std is
    an absolute dollar figure calibrated to the full-portfolio scale (~$60K).
    For a single product predicting ~$15K that produces absurd bands.

    Better approach: use MAPE (mean absolute percentage error) as a relative
    error estimate.  MAPE of 17 % on a $15K prediction → ±$2,550 half-width.
    This scales correctly regardless of product size.

    We use MAPE only when res_std is disproportionately large relative to the
    prediction (i.e. res_std > 50 % of predicted), falling back to the absolute
    res_std when the scales are comparable.
    """
    if predicted <= 0:
        return 0.0

    # MAPE-based half-width: Z_95 × (MAPE/100) × predicted
    # This is the standard "prediction interval from percentage error" formula.
    mape_half = Z_95 * (mape / 100.0) * predicted

    if res_std and res_std > 0:
        # Only use res_std when it's on the same scale as the prediction
        # (i.e. res_std is less than 50 % of predicted value)
        if res_std < 0.5 * predicted:
            return Z_95 * res_std
        # Otherwise fall back to MAPE-based — it scales with the prediction
        return mape_half

    # No res_std available: use MAPE
    if mape and mape > 0:
        return mape_half

    # Last resort fallback
    return predicted * 0.15


def predict_single(
    model_name: str,
    product_id: int,
    year: int,
    month_num: int,
    base_price: float,
    lag_1_revenue: float,
    lag_2_revenue: float,
) -> dict:
    model = _load_model(model_name)
    if model is None:
        return {
            "model": model_name, "predicted": 0.0,
            "lower_bound": 0.0, "upper_bound": 0.0,
            "mape": 0.0, "r2": 0.0,
            "error": f"Model '{model_name}' not found. Run train_sales_model first.",
        }

    try:
        import pandas as pd

        features = pd.DataFrame([{
            "product_id":    product_id,
            "year":          year,
            "month_num":     month_num,
            "base_price":    base_price,
            "lag_1_revenue": lag_1_revenue,
            "lag_2_revenue": lag_2_revenue,
        }])

        predicted = max(0.0, float(model.predict(features)[0]))

        metrics = _load_metrics().get(model_name, {})
        mape    = metrics.get("mape", 15.0)
        res_std = metrics.get("residual_std", 0.0)
        r2      = metrics.get("r2", 0.0)

        half_width  = _ci_half_width(predicted, mape, res_std)
        lower_bound = max(0.0, predicted - half_width)
        upper_bound = predicted + half_width

        return {
            "model":       model_name,
            "predicted":   round(predicted, 2),
            "lower_bound": round(lower_bound, 2),
            "upper_bound": round(upper_bound, 2),
            "mape":        round(mape, 2),
            "r2":          round(r2, 4),
            "error":       None,
        }

    except Exception as exc:
        return {
            "model": model_name, "predicted": 0.0,
            "lower_bound": 0.0, "upper_bound": 0.0,
            "mape": 0.0, "r2": 0.0,
            "error": str(exc),
        }


def predict_horizon(
    model_name: str,
    product_id: int,
    base_price: float,
    lag_1_revenue: float,
    lag_2_revenue: float,
    start_year: int,
    start_month: int,
    horizon_months: int = 12,
) -> list:
    """
    Rolling forecast for `horizon_months` months.
    Each month's prediction feeds the next month's lag features.
    """
    results = []
    lag_1   = lag_1_revenue
    lag_2   = lag_2_revenue
    year    = start_year
    month   = start_month

    for _ in range(horizon_months):
        result = predict_single(
            model_name    = model_name,
            product_id    = product_id,
            year          = year,
            month_num     = month,
            base_price    = base_price,
            lag_1_revenue = lag_1,
            lag_2_revenue = lag_2,
        )

        results.append({
            "month":       f"{year}-{month:02d}",
            "predicted":   result["predicted"],
            "lower_bound": result["lower_bound"],
            "upper_bound": result["upper_bound"],
            "error":       result["error"],
        })

        if result["error"]:
            break

        lag_2  = lag_1
        lag_1  = result["predicted"]
        month += 1
        if month > 12:
            month = 1
            year += 1

    return results


def invalidate_cache():
    """Call after re-training to force models to reload from disk."""
    _MODEL_CACHE.clear()
    _METRICS_CACHE.clear()