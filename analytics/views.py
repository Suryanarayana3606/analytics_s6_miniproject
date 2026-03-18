import csv
import json
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.db.models import Sum, Count, Avg
from django.db.models.functions import TruncMonth
from django.contrib.auth.decorators import login_required
from .models import Customer, Product, Sales_Transaction, Sales_Forecast, FM_Customer_Segment
from .ml.train_model import load_persisted_metrics

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_feature_importances(model_name):
    """
    Return real feature importances from the saved model file.
    All heavy imports are lazy so Django startup is unaffected.
    """
    import os
    try:
        import joblib
        import numpy as np
    except ImportError:
        return None
    from django.conf import settings

    MODEL_DIR = os.path.join(settings.BASE_DIR, 'analytics', 'ml')
    path = os.path.join(MODEL_DIR, f'{model_name.lower()}_model.pkl')
    if not os.path.exists(path):
        return None

    feature_labels = {
        'product_id':     'Product Identity',
        'year':           'Year Trend',
        'month_num':      'Seasonality (Month)',
        'base_price':     'Base Price',
        'lag_1_revenue':  'Lag Revenue (t-1)',
        'lag_2_revenue':  'Lag Revenue (t-2)',
    }
    feature_order = list(feature_labels.keys())

    try:
        model = joblib.load(path)
        if hasattr(model, 'feature_importances_'):
            fi = model.feature_importances_
            pairs = sorted(
                [(feature_labels[k], round(v * 100, 1)) for k, v in zip(feature_order, fi)],
                key=lambda x: -x[1]
            )
            return pairs[:5]   # top-5 drivers
        elif hasattr(model, 'coef_'):
            # Ridge — use absolute normalised coefficients
            import numpy as np
            coef = model.coef_
            abs_coef = np.abs(coef)
            total = abs_coef.sum() or 1
            pairs = sorted(
                [(feature_labels[k], round(v / total * 100, 1)) for k, v in zip(feature_order, abs_coef)],
                key=lambda x: -x[1]
            )
            return pairs[:5]
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

@login_required
def dashboard_view(request):
    industry_filter = request.GET.get('industry', '')
    region_filter   = request.GET.get('region', '')

    transactions = Sales_Transaction.objects.all()
    if industry_filter:
        transactions = transactions.filter(customer__industry=industry_filter)
    if region_filter:
        transactions = transactions.filter(customer__region=region_filter)

    total_revenue   = transactions.aggregate(Sum('revenue'))['revenue__sum'] or 0
    total_customers = transactions.values('customer').distinct().count()
    total_products  = Product.objects.count()

    industry_revenue = list(
        transactions.values('customer__industry')
        .annotate(revenue=Sum('revenue'))
        .order_by('-revenue')
    )
    chart_labels = [item['customer__industry'] for item in industry_revenue]
    chart_data   = [float(item['revenue']) for item in industry_revenue]

    region_revenue = list(
        transactions.values('customer__region')
        .annotate(revenue=Sum('revenue'))
        .order_by('-revenue')
    )
    region_labels = [item['customer__region'] for item in region_revenue]
    region_data   = [float(item['revenue']) for item in region_revenue]

    context = {
        'total_revenue':   total_revenue,
        'total_customers': total_customers,
        'total_products':  total_products,
        'chart_labels':    json.dumps(chart_labels),
        'chart_data':      json.dumps(chart_data),
        'region_labels':   json.dumps(region_labels),
        'region_data':     json.dumps(region_data),
        'industries':      ['Technology', 'Healthcare', 'Finance', 'Education', 'Manufacturing', 'Retail'],
        'regions':         ['North America', 'Europe', 'Asia Pacific', 'Latin America'],
        'selected_industry': industry_filter,
        'selected_region':   region_filter,
    }
    return render(request, 'analytics/dashboard.html', context)


# ---------------------------------------------------------------------------
# Sales Forecast
# ---------------------------------------------------------------------------

@login_required
def sales_forecast_view(request):
    # ── Historical data (last 12 months) ─────────────────────────────────────
    historical_data = list(
        Sales_Transaction.objects
        .annotate(month=TruncMonth('transaction_date'))
        .values('month')
        .annotate(total_revenue=Sum('revenue'))
        .order_by('month')
    )
    if len(historical_data) > 12:
        historical_data = historical_data[-12:]

    hist_dates    = [d['month'].strftime("%Y-%m") for d in historical_data]
    hist_revenues = [float(d['total_revenue']) for d in historical_data]

    # ── Parameters ───────────────────────────────────────────────────────────
    selected_model = request.GET.get('model', 'XGBoost')
    try:
        horizon_months = int(request.GET.get('horizon', '12'))
    except ValueError:
        horizon_months = 12
    horizon_str = str(horizon_months)

    models_list = ['XGBoost', 'Prophet', 'ARIMA', 'LSTM']
    if selected_model not in models_list:
        selected_model = 'XGBoost'

    # ── Load persisted real metrics ───────────────────────────────────────────
    persisted = load_persisted_metrics()

    # ── Build model comparison table ─────────────────────────────────────────
    model_metrics = []
    best_model    = None
    best_mape     = float('inf')

    for model_name in models_list:
        qs = Sales_Forecast.objects.filter(model_version=model_name)

        if qs.exists():
            # Use persisted training metrics (real values); fall back to DB avg
            pm = persisted.get(model_name, {})
            avg_mape = pm.get('mape', float(qs.aggregate(Avg('mape'))['mape__avg'] or 5.0))
            mae      = pm.get('mae',  0.0)
            rmse     = pm.get('rmse', 0.0)
            r2       = pm.get('r2',   0.0)
        else:
            avg_mape = 0.0
            mae = rmse = r2 = 0.0

        model_metrics.append({
            'name': model_name,
            'mape': round(avg_mape, 2),
            'mae':  round(mae, 2),
            'rmse': round(rmse, 2),
            'r2':   round(r2, 4),
        })

        if avg_mape < best_mape and avg_mape > 0:
            best_mape  = avg_mape
            best_model = model_name

    # ── Selected-model forecast data ─────────────────────────────────────────
    model_qs = Sales_Forecast.objects.filter(model_version=selected_model)

    agg_forecasts = list(
        model_qs.values('forecast_date')
        .annotate(
            total_revenue=Sum('forecast_revenue'),
            total_lower=Sum('lower_bound'),
            total_upper=Sum('upper_bound'),
        )
        .order_by('forecast_date')[:horizon_months]
    )

    forecast_dates   = [f['forecast_date'].strftime("%Y-%m") for f in agg_forecasts]
    forecast_revenues = [float(f['total_revenue']) for f in agg_forecasts]
    forecast_lower    = [float(f['total_lower'])   for f in agg_forecasts]
    forecast_upper    = [float(f['total_upper'])   for f in agg_forecasts]

    # Merge timelines: nulls where one series is absent
    all_dates = hist_dates + forecast_dates
    aligned_hist     = hist_revenues + [None] * len(forecast_dates)
    aligned_forecast = [None] * len(hist_revenues) + forecast_revenues
    aligned_lower    = [None] * len(hist_revenues) + forecast_lower
    aligned_upper    = [None] * len(hist_revenues) + forecast_upper

    # Bridge gap: copy last historical point to first forecast position
    if hist_revenues and forecast_revenues:
        bridge = len(hist_revenues) - 1
        aligned_forecast[bridge] = hist_revenues[-1]
        aligned_lower[bridge]    = hist_revenues[-1]
        aligned_upper[bridge]    = hist_revenues[-1]

    # ── Selected model's metrics ──────────────────────────────────────────────
    selected_metrics = next(
        (m for m in model_metrics if m['name'] == selected_model),
        model_metrics[0] if model_metrics else {}
    )
    avg_mape = selected_metrics.get('mape', 0)
    mae      = selected_metrics.get('mae',  0)
    rmse     = selected_metrics.get('rmse', 0)
    r2       = selected_metrics.get('r2',   0)

    avg_revenue = (sum(forecast_revenues) / len(forecast_revenues)) if forecast_revenues else 0

    # ── Top products over the selected horizon ────────────────────────────────
    unique_forecast_dates = list(
        model_qs.values_list('forecast_date', flat=True)
        .distinct().order_by('forecast_date')[:horizon_months]
    )
    top_products = list(
        model_qs.filter(forecast_date__in=unique_forecast_dates)
        .values('product__product_name')
        .annotate(total_forecast=Sum('forecast_revenue'))
        .order_by('-total_forecast')[:5]
    )

    # ── Real feature importances ──────────────────────────────────────────────
    feature_importances = _get_feature_importances(selected_model)

    # Products list for real-time prediction panel
    # Build product list with real lag values from DB for the real-time panel
    
    from collections import defaultdict

    # Get the last 2 months of revenue per product
    recent_revenue = list(
        Sales_Transaction.objects
        .annotate(month=TruncMonth('transaction_date'))
        .values('product_id', 'month')
        .annotate(total=Sum('revenue'))
        .order_by('product_id', '-month')
    )

    # Build {product_id: [most_recent_rev, second_most_recent_rev]}
    product_lags = defaultdict(list)
    for row in recent_revenue:
        pid = row['product_id']
        if len(product_lags[pid]) < 2:
            product_lags[pid].append(float(row['total']))

    all_products = []
    for p in Product.objects.values('product_id', 'product_name', 'base_price').order_by('product_name'):
        lags = product_lags.get(p['product_id'], [0.0, 0.0])
        all_products.append({
            'product_id':   p['product_id'],
            'product_name': p['product_name'],
            'base_price':   float(p['base_price']),
            'lag_1':        round(lags[0], 2) if len(lags) > 0 else 0.0,
            'lag_2':        round(lags[1], 2) if len(lags) > 1 else 0.0,
        })

    context = {
        'selected_model':      selected_model,
        'horizon':             horizon_str,
        'models':              models_list,
        'model_metrics':       model_metrics,
        'best_model':          best_model or selected_model,
        'forecast_dates':      json.dumps(all_dates),
        'historical_revenues': json.dumps(aligned_hist),
        'forecast_revenues':   json.dumps(aligned_forecast),
        'lower_bounds':        json.dumps(aligned_lower),
        'upper_bounds':        json.dumps(aligned_upper),
        'top_products':        top_products,
        'mape':                round(avg_mape, 2),
        'mae':                 round(mae, 2),
        'rmse':                round(rmse, 2),
        'r2':                  round(r2, 4),
        'avg_forecast_revenue': round(avg_revenue, 2),
        'feature_importances': feature_importances,  # list[(label, pct)] or None
        'products':            all_products,
    }
    return render(request, 'analytics/sales_forecast.html', context)


# ---------------------------------------------------------------------------
# Customer Segmentation
# ---------------------------------------------------------------------------

@login_required
def segmentation_view(request):
    segments = FM_Customer_Segment.objects.select_related('customer').all()

    segment_counts = list(
        FM_Customer_Segment.objects.values('segment')
        .annotate(count=Count('segment'))
        .order_by('-count')
    )
    labels = [s['segment'] for s in segment_counts]
    data   = [s['count']   for s in segment_counts]

    color_map = {
        'Champions': '#10b981',
        'Loyal':     '#3b82f6',
        'At Risk':   '#f59e0b',
        'New':       '#8b5cf6',
        'Lost':      '#ef4444',
        'Promising': '#06b6d4',
    }

    groups = {}
    for seg in segments:
        s_name = seg.segment
        groups.setdefault(s_name, []).append({
            'x': float(seg.recency),
            'y': float(seg.monetary),
        })

    scatter_datasets = [
        {
            'label':           s_name,
            'data':            points,
            'backgroundColor': color_map.get(s_name, '#06b6d4'),
            'borderColor':     color_map.get(s_name, '#06b6d4'),
            'pointRadius':     5,
            'pointHoverRadius': 8,
        }
        for s_name, points in groups.items()
    ]

    context = {
        'segments':         segments,
        'chart_labels':     json.dumps(labels),
        'chart_data':       json.dumps(data),
        'scatter_datasets': json.dumps(scatter_datasets),
    }
    return render(request, 'analytics/segmentation.html', context)


# ---------------------------------------------------------------------------
# CSV Exports
# ---------------------------------------------------------------------------

@login_required
def export_report_view(request):
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="sales_report.csv"'
    writer = csv.writer(response)
    writer.writerow(['Transaction ID', 'Date', 'Customer', 'Industry', 'Product', 'Quantity', 'Revenue'])
    for txn in Sales_Transaction.objects.select_related('customer', 'product').all():
        writer.writerow([
            txn.transaction_id, txn.transaction_date,
            txn.customer.customer_name, txn.customer.industry,
            txn.product.product_name, txn.quantity, txn.revenue,
        ])
    return response


@login_required
def export_forecast_csv(request):
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="sales_forecast_export.csv"'
    writer = csv.writer(response)
    writer.writerow(['Forecast Date', 'Product', 'Model', 'Predicted Revenue', 'Lower Bound', 'Upper Bound', 'MAPE (%)'])
    for f in Sales_Forecast.objects.select_related('product').order_by('model_version', 'forecast_date'):
        writer.writerow([
            f.forecast_date, f.product.product_name, f.model_version,
            f.forecast_revenue, f.lower_bound, f.upper_bound, f.mape,
        ])
    return response


# ---------------------------------------------------------------------------
# JSON API
# ---------------------------------------------------------------------------

@login_required
def api_forecast_view(request):
    """
    Serves forecast data as JSON for downstream consumption.
    Query params: ?model=XGBoost  (default)  &horizon=12
    """
    selected_model = request.GET.get('model', 'XGBoost')
    try:
        horizon = int(request.GET.get('horizon', '24'))
    except ValueError:
        horizon = 24

    forecasts = (
        Sales_Forecast.objects
        .filter(model_version=selected_model)
        .order_by('forecast_date')[:horizon]
    )

    metrics = load_persisted_metrics().get(selected_model, {})

    data = [
        {
            'date':              f.forecast_date.strftime('%Y-%m-%d'),
            'product':           f.product.product_name,
            'predicted_revenue': float(f.forecast_revenue),
            'lower_bound':       float(f.lower_bound),
            'upper_bound':       float(f.upper_bound),
        }
        for f in forecasts
    ]

    return JsonResponse({
        'status': 'success',
        'meta': {
            'model':               selected_model,
            'horizon_months':      horizon,
            'mape_pct':            metrics.get('mape'),
            'r2':                  metrics.get('r2'),
            'confidence_interval': '95%',
        },
        'data': data,
    })


# ---------------------------------------------------------------------------
# Real-time Prediction API
# ---------------------------------------------------------------------------

@login_required
def api_realtime_predict(request):
    """
    Real-time prediction endpoint — runs model.predict() live on every call.
    No DB read/write.  Models are cached in-process after first load.

    POST body (JSON):
    {
        "model":           "XGBoost",          // required
        "product_id":      1,                  // required
        "base_price":      1200.0,             // required
        "lag_1_revenue":   15000.0,            // required
        "lag_2_revenue":   14200.0,            // required
        "start_year":      2025,               // required
        "start_month":     10,                 // required
        "horizon_months":  12                  // optional, default 12
    }

    Response:
    {
        "status":  "success" | "error",
        "model":   str,
        "horizon": int,
        "data": [{"month": "2025-10", "predicted": ..., "lower_bound": ..., "upper_bound": ...}]
    }
    """
    if request.method == 'GET':
        return JsonResponse({
            'endpoint':     '/api/predict/',
            'method':       'POST',
            'description':  'Real-time forecast — runs model.predict() live, no DB involved.',
            'content_type': 'application/json',
            'fields': {
                'model':          'XGBoost | Prophet | ARIMA | LSTM  (required)',
                'product_id':     'int  (required)',
                'base_price':     'float  (required)',
                'lag_1_revenue':  'float — last month revenue  (required)',
                'lag_2_revenue':  'float — two months ago  (required)',
                'start_year':     'int  (required)',
                'start_month':    'int 1-12  (required)',
                'horizon_months': 'int 1-36  (optional, default 12)',
            },
            'example': {
                'model': 'XGBoost', 'product_id': 1,
                'base_price': 1200.0, 'lag_1_revenue': 15000.0,
                'lag_2_revenue': 14200.0, 'start_year': 2025,
                'start_month': 10, 'horizon_months': 12,
            },
        })

    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'POST required'}, status=405)

    try:
        body = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({'status': 'error', 'message': 'Invalid JSON body'}, status=400)

    # ── Validate required fields ──────────────────────────────────────────────
    required = ['model', 'product_id', 'base_price', 'lag_1_revenue',
                'lag_2_revenue', 'start_year', 'start_month']
    missing  = [f for f in required if f not in body]
    if missing:
        return JsonResponse({
            'status': 'error',
            'message': f'Missing fields: {", ".join(missing)}'
        }, status=400)

    valid_models = ['XGBoost', 'Prophet', 'ARIMA', 'LSTM']
    model_name   = body.get('model', 'XGBoost')
    if model_name not in valid_models:
        return JsonResponse({
            'status': 'error',
            'message': f'Unknown model. Choose from: {valid_models}'
        }, status=400)

    try:
        product_id      = int(body['product_id'])
        base_price      = float(body['base_price'])
        lag_1_revenue   = float(body['lag_1_revenue'])
        lag_2_revenue   = float(body['lag_2_revenue'])
        start_year      = int(body['start_year'])
        start_month     = int(body['start_month'])
        horizon_months  = int(body.get('horizon_months', 12))
        horizon_months  = max(1, min(horizon_months, 36))  # clamp 1–36
    except (TypeError, ValueError) as exc:
        return JsonResponse({'status': 'error', 'message': f'Type error: {exc}'}, status=400)

    # ── Run live prediction ───────────────────────────────────────────────────
    from analytics.ml.realtime_predict import predict_horizon, _load_metrics

    results = predict_horizon(
        model_name      = model_name,
        product_id      = product_id,
        base_price      = base_price,
        lag_1_revenue   = lag_1_revenue,
        lag_2_revenue   = lag_2_revenue,
        start_year      = start_year,
        start_month     = start_month,
        horizon_months  = horizon_months,
    )

    # Surface any model-load error as a top-level error response
    if results and results[0].get('error'):
        return JsonResponse({
            'status':  'error',
            'message': results[0]['error'],
        }, status=500)

    metrics = _load_metrics().get(model_name, {})

    return JsonResponse({
        'status':  'success',
        'model':   model_name,
        'horizon': horizon_months,
        'meta': {
            'mape': metrics.get('mape'),
            'r2':   metrics.get('r2'),
            'confidence_interval': '95%',
        },
        'data': results,
    })
'''
v3
import csv
import json
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.db.models import Sum, Count, Avg
from django.db.models.functions import TruncMonth
from django.contrib.auth.decorators import login_required
from .models import Customer, Product, Sales_Transaction, Sales_Forecast, FM_Customer_Segment
from .ml.train_model import load_persisted_metrics

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_feature_importances(model_name):
    """
    Return real feature importances from the saved model file.
    All heavy imports are lazy so Django startup is unaffected.
    """
    import os
    try:
        import joblib
        import numpy as np
    except ImportError:
        return None
    from django.conf import settings

    MODEL_DIR = os.path.join(settings.BASE_DIR, 'analytics', 'ml')
    path = os.path.join(MODEL_DIR, f'{model_name.lower()}_model.pkl')
    if not os.path.exists(path):
        return None

    feature_labels = {
        'product_id':     'Product Identity',
        'year':           'Year Trend',
        'month_num':      'Seasonality (Month)',
        'base_price':     'Base Price',
        'lag_1_revenue':  'Lag Revenue (t-1)',
        'lag_2_revenue':  'Lag Revenue (t-2)',
    }
    feature_order = list(feature_labels.keys())

    try:
        model = joblib.load(path)
        if hasattr(model, 'feature_importances_'):
            fi = model.feature_importances_
            pairs = sorted(
                [(feature_labels[k], round(v * 100, 1)) for k, v in zip(feature_order, fi)],
                key=lambda x: -x[1]
            )
            return pairs[:5]   # top-5 drivers
        elif hasattr(model, 'coef_'):
            # Ridge — use absolute normalised coefficients
            import numpy as np
            coef = model.coef_
            abs_coef = np.abs(coef)
            total = abs_coef.sum() or 1
            pairs = sorted(
                [(feature_labels[k], round(v / total * 100, 1)) for k, v in zip(feature_order, abs_coef)],
                key=lambda x: -x[1]
            )
            return pairs[:5]
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

@login_required
def dashboard_view(request):
    industry_filter = request.GET.get('industry', '')
    region_filter   = request.GET.get('region', '')

    transactions = Sales_Transaction.objects.all()
    if industry_filter:
        transactions = transactions.filter(customer__industry=industry_filter)
    if region_filter:
        transactions = transactions.filter(customer__region=region_filter)

    total_revenue   = transactions.aggregate(Sum('revenue'))['revenue__sum'] or 0
    total_customers = transactions.values('customer').distinct().count()
    total_products  = Product.objects.count()

    industry_revenue = list(
        transactions.values('customer__industry')
        .annotate(revenue=Sum('revenue'))
        .order_by('-revenue')
    )
    chart_labels = [item['customer__industry'] for item in industry_revenue]
    chart_data   = [float(item['revenue']) for item in industry_revenue]

    region_revenue = list(
        transactions.values('customer__region')
        .annotate(revenue=Sum('revenue'))
        .order_by('-revenue')
    )
    region_labels = [item['customer__region'] for item in region_revenue]
    region_data   = [float(item['revenue']) for item in region_revenue]

    context = {
        'total_revenue':   total_revenue,
        'total_customers': total_customers,
        'total_products':  total_products,
        'chart_labels':    json.dumps(chart_labels),
        'chart_data':      json.dumps(chart_data),
        'region_labels':   json.dumps(region_labels),
        'region_data':     json.dumps(region_data),
        'industries':      ['Technology', 'Healthcare', 'Finance', 'Education', 'Manufacturing', 'Retail'],
        'regions':         ['North America', 'Europe', 'Asia Pacific', 'Latin America'],
        'selected_industry': industry_filter,
        'selected_region':   region_filter,
    }
    return render(request, 'analytics/dashboard.html', context)


# ---------------------------------------------------------------------------
# Sales Forecast
# ---------------------------------------------------------------------------

@login_required
def sales_forecast_view(request):
    # ── Historical data (last 12 months) ─────────────────────────────────────
    historical_data = list(
        Sales_Transaction.objects
        .annotate(month=TruncMonth('transaction_date'))
        .values('month')
        .annotate(total_revenue=Sum('revenue'))
        .order_by('month')
    )
    if len(historical_data) > 12:
        historical_data = historical_data[-12:]

    hist_dates    = [d['month'].strftime("%Y-%m") for d in historical_data]
    hist_revenues = [float(d['total_revenue']) for d in historical_data]

    # ── Parameters ───────────────────────────────────────────────────────────
    selected_model = request.GET.get('model', 'XGBoost')
    try:
        horizon_months = int(request.GET.get('horizon', '12'))
    except ValueError:
        horizon_months = 12
    horizon_str = str(horizon_months)

    models_list = ['XGBoost', 'Prophet', 'ARIMA', 'LSTM']
    if selected_model not in models_list:
        selected_model = 'XGBoost'

    # ── Load persisted real metrics ───────────────────────────────────────────
    persisted = load_persisted_metrics()

    # ── Build model comparison table ─────────────────────────────────────────
    model_metrics = []
    best_model    = None
    best_mape     = float('inf')

    for model_name in models_list:
        qs = Sales_Forecast.objects.filter(model_version=model_name)

        if qs.exists():
            # Use persisted training metrics (real values); fall back to DB avg
            pm = persisted.get(model_name, {})
            avg_mape = pm.get('mape', float(qs.aggregate(Avg('mape'))['mape__avg'] or 5.0))
            mae      = pm.get('mae',  0.0)
            rmse     = pm.get('rmse', 0.0)
            r2       = pm.get('r2',   0.0)
        else:
            avg_mape = 0.0
            mae = rmse = r2 = 0.0

        model_metrics.append({
            'name': model_name,
            'mape': round(avg_mape, 2),
            'mae':  round(mae, 2),
            'rmse': round(rmse, 2),
            'r2':   round(r2, 4),
        })

        if avg_mape < best_mape and avg_mape > 0:
            best_mape  = avg_mape
            best_model = model_name

    # ── Selected-model forecast data ─────────────────────────────────────────
    model_qs = Sales_Forecast.objects.filter(model_version=selected_model)

    agg_forecasts = list(
        model_qs.values('forecast_date')
        .annotate(
            total_revenue=Sum('forecast_revenue'),
            total_lower=Sum('lower_bound'),
            total_upper=Sum('upper_bound'),
        )
        .order_by('forecast_date')[:horizon_months]
    )

    forecast_dates   = [f['forecast_date'].strftime("%Y-%m") for f in agg_forecasts]
    forecast_revenues = [float(f['total_revenue']) for f in agg_forecasts]
    forecast_lower    = [float(f['total_lower'])   for f in agg_forecasts]
    forecast_upper    = [float(f['total_upper'])   for f in agg_forecasts]

    # Merge timelines: nulls where one series is absent
    all_dates = hist_dates + forecast_dates
    aligned_hist     = hist_revenues + [None] * len(forecast_dates)
    aligned_forecast = [None] * len(hist_revenues) + forecast_revenues
    aligned_lower    = [None] * len(hist_revenues) + forecast_lower
    aligned_upper    = [None] * len(hist_revenues) + forecast_upper

    # Bridge gap: copy last historical point to first forecast position
    if hist_revenues and forecast_revenues:
        bridge = len(hist_revenues) - 1
        aligned_forecast[bridge] = hist_revenues[-1]
        aligned_lower[bridge]    = hist_revenues[-1]
        aligned_upper[bridge]    = hist_revenues[-1]

    # ── Selected model's metrics ──────────────────────────────────────────────
    selected_metrics = next(
        (m for m in model_metrics if m['name'] == selected_model),
        model_metrics[0] if model_metrics else {}
    )
    avg_mape = selected_metrics.get('mape', 0)
    mae      = selected_metrics.get('mae',  0)
    rmse     = selected_metrics.get('rmse', 0)
    r2       = selected_metrics.get('r2',   0)

    avg_revenue = (sum(forecast_revenues) / len(forecast_revenues)) if forecast_revenues else 0

    # ── Top products over the selected horizon ────────────────────────────────
    unique_forecast_dates = list(
        model_qs.values_list('forecast_date', flat=True)
        .distinct().order_by('forecast_date')[:horizon_months]
    )
    top_products = list(
        model_qs.filter(forecast_date__in=unique_forecast_dates)
        .values('product__product_name')
        .annotate(total_forecast=Sum('forecast_revenue'))
        .order_by('-total_forecast')[:5]
    )

    # ── Real feature importances ──────────────────────────────────────────────
    feature_importances = _get_feature_importances(selected_model)

    # Products list for real-time prediction panel
    all_products = list(Product.objects.values("product_id", "product_name", "base_price").order_by("product_name"))

    context = {
        'selected_model':      selected_model,
        'horizon':             horizon_str,
        'models':              models_list,
        'model_metrics':       model_metrics,
        'best_model':          best_model or selected_model,
        'forecast_dates':      json.dumps(all_dates),
        'historical_revenues': json.dumps(aligned_hist),
        'forecast_revenues':   json.dumps(aligned_forecast),
        'lower_bounds':        json.dumps(aligned_lower),
        'upper_bounds':        json.dumps(aligned_upper),
        'top_products':        top_products,
        'mape':                round(avg_mape, 2),
        'mae':                 round(mae, 2),
        'rmse':                round(rmse, 2),
        'r2':                  round(r2, 4),
        'avg_forecast_revenue': round(avg_revenue, 2),
        'feature_importances': feature_importances,  # list[(label, pct)] or None
        'products':            all_products,
    }
    return render(request, 'analytics/sales_forecast.html', context)


# ---------------------------------------------------------------------------
# Customer Segmentation
# ---------------------------------------------------------------------------

@login_required
def segmentation_view(request):
    segments = FM_Customer_Segment.objects.select_related('customer').all()

    segment_counts = list(
        FM_Customer_Segment.objects.values('segment')
        .annotate(count=Count('segment'))
        .order_by('-count')
    )
    labels = [s['segment'] for s in segment_counts]
    data   = [s['count']   for s in segment_counts]

    color_map = {
        'Champions': '#10b981',
        'Loyal':     '#3b82f6',
        'At Risk':   '#f59e0b',
        'New':       '#8b5cf6',
        'Lost':      '#ef4444',
        'Promising': '#06b6d4',
    }

    groups = {}
    for seg in segments:
        s_name = seg.segment
        groups.setdefault(s_name, []).append({
            'x': float(seg.recency),
            'y': float(seg.monetary),
        })

    scatter_datasets = [
        {
            'label':           s_name,
            'data':            points,
            'backgroundColor': color_map.get(s_name, '#06b6d4'),
            'borderColor':     color_map.get(s_name, '#06b6d4'),
            'pointRadius':     5,
            'pointHoverRadius': 8,
        }
        for s_name, points in groups.items()
    ]

    context = {
        'segments':         segments,
        'chart_labels':     json.dumps(labels),
        'chart_data':       json.dumps(data),
        'scatter_datasets': json.dumps(scatter_datasets),
    }
    return render(request, 'analytics/segmentation.html', context)


# ---------------------------------------------------------------------------
# CSV Exports
# ---------------------------------------------------------------------------

@login_required
def export_report_view(request):
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="sales_report.csv"'
    writer = csv.writer(response)
    writer.writerow(['Transaction ID', 'Date', 'Customer', 'Industry', 'Product', 'Quantity', 'Revenue'])
    for txn in Sales_Transaction.objects.select_related('customer', 'product').all():
        writer.writerow([
            txn.transaction_id, txn.transaction_date,
            txn.customer.customer_name, txn.customer.industry,
            txn.product.product_name, txn.quantity, txn.revenue,
        ])
    return response


@login_required
def export_forecast_csv(request):
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="sales_forecast_export.csv"'
    writer = csv.writer(response)
    writer.writerow(['Forecast Date', 'Product', 'Model', 'Predicted Revenue', 'Lower Bound', 'Upper Bound', 'MAPE (%)'])
    for f in Sales_Forecast.objects.select_related('product').order_by('model_version', 'forecast_date'):
        writer.writerow([
            f.forecast_date, f.product.product_name, f.model_version,
            f.forecast_revenue, f.lower_bound, f.upper_bound, f.mape,
        ])
    return response


# ---------------------------------------------------------------------------
# JSON API
# ---------------------------------------------------------------------------

@login_required
def api_forecast_view(request):
    """
    Serves forecast data as JSON for downstream consumption.
    Query params: ?model=XGBoost  (default)  &horizon=12
    """
    selected_model = request.GET.get('model', 'XGBoost')
    try:
        horizon = int(request.GET.get('horizon', '24'))
    except ValueError:
        horizon = 24

    forecasts = (
        Sales_Forecast.objects
        .filter(model_version=selected_model)
        .order_by('forecast_date')[:horizon]
    )

    metrics = load_persisted_metrics().get(selected_model, {})

    data = [
        {
            'date':              f.forecast_date.strftime('%Y-%m-%d'),
            'product':           f.product.product_name,
            'predicted_revenue': float(f.forecast_revenue),
            'lower_bound':       float(f.lower_bound),
            'upper_bound':       float(f.upper_bound),
        }
        for f in forecasts
    ]

    return JsonResponse({
        'status': 'success',
        'meta': {
            'model':               selected_model,
            'horizon_months':      horizon,
            'mape_pct':            metrics.get('mape'),
            'r2':                  metrics.get('r2'),
            'confidence_interval': '95%',
        },
        'data': data,
    })


# ---------------------------------------------------------------------------
# Real-time Prediction API
# ---------------------------------------------------------------------------

@login_required
def api_realtime_predict(request):
    """
    Real-time prediction endpoint — runs model.predict() live on every call.
    No DB read/write.  Models are cached in-process after first load.

    POST body (JSON):
    {
        "model":           "XGBoost",          // required
        "product_id":      1,                  // required
        "base_price":      1200.0,             // required
        "lag_1_revenue":   15000.0,            // required
        "lag_2_revenue":   14200.0,            // required
        "start_year":      2025,               // required
        "start_month":     10,                 // required
        "horizon_months":  12                  // optional, default 12
    }

    Response:
    {
        "status":  "success" | "error",
        "model":   str,
        "horizon": int,
        "data": [{"month": "2025-10", "predicted": ..., "lower_bound": ..., "upper_bound": ...}]
    }
    """
    if request.method == 'GET':
        return JsonResponse({
            'endpoint':     '/api/predict/',
            'method':       'POST',
            'description':  'Real-time forecast — runs model.predict() live, no DB involved.',
            'content_type': 'application/json',
            'fields': {
                'model':          'XGBoost | Prophet | ARIMA | LSTM  (required)',
                'product_id':     'int  (required)',
                'base_price':     'float  (required)',
                'lag_1_revenue':  'float — last month revenue  (required)',
                'lag_2_revenue':  'float — two months ago  (required)',
                'start_year':     'int  (required)',
                'start_month':    'int 1-12  (required)',
                'horizon_months': 'int 1-36  (optional, default 12)',
            },
            'example': {
                'model': 'XGBoost', 'product_id': 1,
                'base_price': 1200.0, 'lag_1_revenue': 15000.0,
                'lag_2_revenue': 14200.0, 'start_year': 2025,
                'start_month': 10, 'horizon_months': 12,
            },
        })

    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'POST required'}, status=405)

    try:
        body = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({'status': 'error', 'message': 'Invalid JSON body'}, status=400)

    # ── Validate required fields ──────────────────────────────────────────────
    required = ['model', 'product_id', 'base_price', 'lag_1_revenue',
                'lag_2_revenue', 'start_year', 'start_month']
    missing  = [f for f in required if f not in body]
    if missing:
        return JsonResponse({
            'status': 'error',
            'message': f'Missing fields: {", ".join(missing)}'
        }, status=400)

    valid_models = ['XGBoost', 'Prophet', 'ARIMA', 'LSTM']
    model_name   = body.get('model', 'XGBoost')
    if model_name not in valid_models:
        return JsonResponse({
            'status': 'error',
            'message': f'Unknown model. Choose from: {valid_models}'
        }, status=400)

    try:
        product_id      = int(body['product_id'])
        base_price      = float(body['base_price'])
        lag_1_revenue   = float(body['lag_1_revenue'])
        lag_2_revenue   = float(body['lag_2_revenue'])
        start_year      = int(body['start_year'])
        start_month     = int(body['start_month'])
        horizon_months  = int(body.get('horizon_months', 12))
        horizon_months  = max(1, min(horizon_months, 36))  # clamp 1–36
    except (TypeError, ValueError) as exc:
        return JsonResponse({'status': 'error', 'message': f'Type error: {exc}'}, status=400)

    # ── Run live prediction ───────────────────────────────────────────────────
    from analytics.ml.realtime_predict import predict_horizon, _load_metrics

    results = predict_horizon(
        model_name      = model_name,
        product_id      = product_id,
        base_price      = base_price,
        lag_1_revenue   = lag_1_revenue,
        lag_2_revenue   = lag_2_revenue,
        start_year      = start_year,
        start_month     = start_month,
        horizon_months  = horizon_months,
    )

    # Surface any model-load error as a top-level error response
    if results and results[0].get('error'):
        return JsonResponse({
            'status':  'error',
            'message': results[0]['error'],
        }, status=500)

    metrics = _load_metrics().get(model_name, {})

    return JsonResponse({
        'status':  'success',
        'model':   model_name,
        'horizon': horizon_months,
        'meta': {
            'mape': metrics.get('mape'),
            'r2':   metrics.get('r2'),
            'confidence_interval': '95%',
        },
        'data': results,
    })

------
v2
import csv
import json
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.db.models import Sum, Count, Avg
from django.db.models.functions import TruncMonth
from django.contrib.auth.decorators import login_required
from .models import Customer, Product, Sales_Transaction, Sales_Forecast, FM_Customer_Segment
from .ml.train_model import load_persisted_metrics

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_feature_importances(model_name):
    """
    Return real feature importances from the saved model file.
    All heavy imports are lazy so Django startup is unaffected.
    """
    import os
    try:
        import joblib
        import numpy as np
    except ImportError:
        return None
    from django.conf import settings

    MODEL_DIR = os.path.join(settings.BASE_DIR, 'analytics', 'ml')
    path = os.path.join(MODEL_DIR, f'{model_name.lower()}_model.pkl')
    if not os.path.exists(path):
        return None

    feature_labels = {
        'product_id':     'Product Identity',
        'year':           'Year Trend',
        'month_num':      'Seasonality (Month)',
        'base_price':     'Base Price',
        'lag_1_revenue':  'Lag Revenue (t-1)',
        'lag_2_revenue':  'Lag Revenue (t-2)',
    }
    feature_order = list(feature_labels.keys())

    try:
        model = joblib.load(path)
        if hasattr(model, 'feature_importances_'):
            fi = model.feature_importances_
            pairs = sorted(
                [(feature_labels[k], round(v * 100, 1)) for k, v in zip(feature_order, fi)],
                key=lambda x: -x[1]
            )
            return pairs[:5]   # top-5 drivers
        elif hasattr(model, 'coef_'):
            # Ridge — use absolute normalised coefficients
            import numpy as np
            coef = model.coef_
            abs_coef = np.abs(coef)
            total = abs_coef.sum() or 1
            pairs = sorted(
                [(feature_labels[k], round(v / total * 100, 1)) for k, v in zip(feature_order, abs_coef)],
                key=lambda x: -x[1]
            )
            return pairs[:5]
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

#@login_required
def dashboard_view(request):
    industry_filter = request.GET.get('industry', '')
    region_filter   = request.GET.get('region', '')

    transactions = Sales_Transaction.objects.all()
    if industry_filter:
        transactions = transactions.filter(customer__industry=industry_filter)
    if region_filter:
        transactions = transactions.filter(customer__region=region_filter)

    total_revenue   = transactions.aggregate(Sum('revenue'))['revenue__sum'] or 0
    total_customers = transactions.values('customer').distinct().count()
    total_products  = Product.objects.count()

    industry_revenue = list(
        transactions.values('customer__industry')
        .annotate(revenue=Sum('revenue'))
        .order_by('-revenue')
    )
    chart_labels = [item['customer__industry'] for item in industry_revenue]
    chart_data   = [float(item['revenue']) for item in industry_revenue]

    region_revenue = list(
        transactions.values('customer__region')
        .annotate(revenue=Sum('revenue'))
        .order_by('-revenue')
    )
    region_labels = [item['customer__region'] for item in region_revenue]
    region_data   = [float(item['revenue']) for item in region_revenue]

    context = {
        'total_revenue':   total_revenue,
        'total_customers': total_customers,
        'total_products':  total_products,
        'chart_labels':    json.dumps(chart_labels),
        'chart_data':      json.dumps(chart_data),
        'region_labels':   json.dumps(region_labels),
        'region_data':     json.dumps(region_data),
        'industries':      ['Technology', 'Healthcare', 'Finance', 'Education', 'Manufacturing', 'Retail'],
        'regions':         ['North America', 'Europe', 'Asia Pacific', 'Latin America'],
        'selected_industry': industry_filter,
        'selected_region':   region_filter,
    }
    return render(request, 'analytics/dashboard.html', context)


# ---------------------------------------------------------------------------
# Sales Forecast
# ---------------------------------------------------------------------------

#@login_required
def sales_forecast_view(request):
    # ── Historical data (last 12 months) ─────────────────────────────────────
    historical_data = list(
        Sales_Transaction.objects
        .annotate(month=TruncMonth('transaction_date'))
        .values('month')
        .annotate(total_revenue=Sum('revenue'))
        .order_by('month')
    )
    if len(historical_data) > 12:
        historical_data = historical_data[-12:]

    hist_dates    = [d['month'].strftime("%Y-%m") for d in historical_data]
    hist_revenues = [float(d['total_revenue']) for d in historical_data]

    # ── Parameters ───────────────────────────────────────────────────────────
    selected_model = request.GET.get('model', 'XGBoost')
    try:
        horizon_months = int(request.GET.get('horizon', '12'))
    except ValueError:
        horizon_months = 12
    horizon_str = str(horizon_months)

    models_list = ['XGBoost', 'Prophet', 'ARIMA', 'LSTM']
    if selected_model not in models_list:
        selected_model = 'XGBoost'

    # ── Load persisted real metrics ───────────────────────────────────────────
    persisted = load_persisted_metrics()

    # ── Build model comparison table ─────────────────────────────────────────
    model_metrics = []
    best_model    = None
    best_mape     = float('inf')

    for model_name in models_list:
        qs = Sales_Forecast.objects.filter(model_version=model_name)

        if qs.exists():
            # Use persisted training metrics (real values); fall back to DB avg
            pm = persisted.get(model_name, {})
            avg_mape = pm.get('mape', float(qs.aggregate(Avg('mape'))['mape__avg'] or 5.0))
            mae      = pm.get('mae',  0.0)
            rmse     = pm.get('rmse', 0.0)
            r2       = pm.get('r2',   0.0)
        else:
            avg_mape = 0.0
            mae = rmse = r2 = 0.0

        model_metrics.append({
            'name': model_name,
            'mape': round(avg_mape, 2),
            'mae':  round(mae, 2),
            'rmse': round(rmse, 2),
            'r2':   round(r2, 4),
        })

        if avg_mape < best_mape and avg_mape > 0:
            best_mape  = avg_mape
            best_model = model_name

    # ── Selected-model forecast data ─────────────────────────────────────────
    model_qs = Sales_Forecast.objects.filter(model_version=selected_model)

    agg_forecasts = list(
        model_qs.values('forecast_date')
        .annotate(
            total_revenue=Sum('forecast_revenue'),
            total_lower=Sum('lower_bound'),
            total_upper=Sum('upper_bound'),
        )
        .order_by('forecast_date')[:horizon_months]
    )

    forecast_dates   = [f['forecast_date'].strftime("%Y-%m") for f in agg_forecasts]
    forecast_revenues = [float(f['total_revenue']) for f in agg_forecasts]
    forecast_lower    = [float(f['total_lower'])   for f in agg_forecasts]
    forecast_upper    = [float(f['total_upper'])   for f in agg_forecasts]

    # Merge timelines: nulls where one series is absent
    all_dates = hist_dates + forecast_dates
    aligned_hist     = hist_revenues + [None] * len(forecast_dates)
    aligned_forecast = [None] * len(hist_revenues) + forecast_revenues
    aligned_lower    = [None] * len(hist_revenues) + forecast_lower
    aligned_upper    = [None] * len(hist_revenues) + forecast_upper

    # Bridge gap: copy last historical point to first forecast position
    if hist_revenues and forecast_revenues:
        bridge = len(hist_revenues) - 1
        aligned_forecast[bridge] = hist_revenues[-1]
        aligned_lower[bridge]    = hist_revenues[-1]
        aligned_upper[bridge]    = hist_revenues[-1]

    # ── Selected model's metrics ──────────────────────────────────────────────
    selected_metrics = next(
        (m for m in model_metrics if m['name'] == selected_model),
        model_metrics[0] if model_metrics else {}
    )
    avg_mape = selected_metrics.get('mape', 0)
    mae      = selected_metrics.get('mae',  0)
    rmse     = selected_metrics.get('rmse', 0)
    r2       = selected_metrics.get('r2',   0)

    avg_revenue = (sum(forecast_revenues) / len(forecast_revenues)) if forecast_revenues else 0

    # ── Top products over the selected horizon ────────────────────────────────
    unique_forecast_dates = list(
        model_qs.values_list('forecast_date', flat=True)
        .distinct().order_by('forecast_date')[:horizon_months]
    )
    top_products = list(
        model_qs.filter(forecast_date__in=unique_forecast_dates)
        .values('product__product_name')
        .annotate(total_forecast=Sum('forecast_revenue'))
        .order_by('-total_forecast')[:5]
    )

    # ── Real feature importances ──────────────────────────────────────────────
    feature_importances = _get_feature_importances(selected_model)

    # Products list for real-time prediction panel
    all_products = list(Product.objects.values("product_id", "product_name", "base_price").order_by("product_name"))

    context = {
        'selected_model':      selected_model,
        'horizon':             horizon_str,
        'models':              models_list,
        'model_metrics':       model_metrics,
        'best_model':          best_model or selected_model,
        'forecast_dates':      json.dumps(all_dates),
        'historical_revenues': json.dumps(aligned_hist),
        'forecast_revenues':   json.dumps(aligned_forecast),
        'lower_bounds':        json.dumps(aligned_lower),
        'upper_bounds':        json.dumps(aligned_upper),
        'top_products':        top_products,
        'mape':                round(avg_mape, 2),
        'mae':                 round(mae, 2),
        'rmse':                round(rmse, 2),
        'r2':                  round(r2, 4),
        'avg_forecast_revenue': round(avg_revenue, 2),
        'feature_importances': feature_importances,  # list[(label, pct)] or None
        'products':            all_products,
    }
    return render(request, 'analytics/sales_forecast.html', context)


# ---------------------------------------------------------------------------
# Customer Segmentation
# ---------------------------------------------------------------------------

#@login_required
def segmentation_view(request):
    segments = FM_Customer_Segment.objects.select_related('customer').all()

    segment_counts = list(
        FM_Customer_Segment.objects.values('segment')
        .annotate(count=Count('segment'))
        .order_by('-count')
    )
    labels = [s['segment'] for s in segment_counts]
    data   = [s['count']   for s in segment_counts]

    color_map = {
        'Champions': '#10b981',
        'Loyal':     '#3b82f6',
        'At Risk':   '#f59e0b',
        'New':       '#8b5cf6',
        'Lost':      '#ef4444',
        'Promising': '#06b6d4',
    }

    groups = {}
    for seg in segments:
        s_name = seg.segment
        groups.setdefault(s_name, []).append({
            'x': float(seg.recency),
            'y': float(seg.monetary),
        })

    scatter_datasets = [
        {
            'label':           s_name,
            'data':            points,
            'backgroundColor': color_map.get(s_name, '#06b6d4'),
            'borderColor':     color_map.get(s_name, '#06b6d4'),
            'pointRadius':     5,
            'pointHoverRadius': 8,
        }
        for s_name, points in groups.items()
    ]

    context = {
        'segments':         segments,
        'chart_labels':     json.dumps(labels),
        'chart_data':       json.dumps(data),
        'scatter_datasets': json.dumps(scatter_datasets),
    }
    return render(request, 'analytics/segmentation.html', context)


# ---------------------------------------------------------------------------
# CSV Exports
# ---------------------------------------------------------------------------

#@login_required
def export_report_view(request):
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="sales_report.csv"'
    writer = csv.writer(response)
    writer.writerow(['Transaction ID', 'Date', 'Customer', 'Industry', 'Product', 'Quantity', 'Revenue'])
    for txn in Sales_Transaction.objects.select_related('customer', 'product').all():
        writer.writerow([
            txn.transaction_id, txn.transaction_date,
            txn.customer.customer_name, txn.customer.industry,
            txn.product.product_name, txn.quantity, txn.revenue,
        ])
    return response


#@login_required
def export_forecast_csv(request):
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="sales_forecast_export.csv"'
    writer = csv.writer(response)
    writer.writerow(['Forecast Date', 'Product', 'Model', 'Predicted Revenue', 'Lower Bound', 'Upper Bound', 'MAPE (%)'])
    for f in Sales_Forecast.objects.select_related('product').order_by('model_version', 'forecast_date'):
        writer.writerow([
            f.forecast_date, f.product.product_name, f.model_version,
            f.forecast_revenue, f.lower_bound, f.upper_bound, f.mape,
        ])
    return response


# ---------------------------------------------------------------------------
# JSON API
# ---------------------------------------------------------------------------

#@login_required
def api_forecast_view(request):
    """
    Serves forecast data as JSON for downstream consumption.
    Query params: ?model=XGBoost  (default)  &horizon=12
    """
    selected_model = request.GET.get('model', 'XGBoost')
    try:
        horizon = int(request.GET.get('horizon', '24'))
    except ValueError:
        horizon = 24

    forecasts = (
        Sales_Forecast.objects
        .filter(model_version=selected_model)
        .order_by('forecast_date')[:horizon]
    )

    metrics = load_persisted_metrics().get(selected_model, {})

    data = [
        {
            'date':              f.forecast_date.strftime('%Y-%m-%d'),
            'product':           f.product.product_name,
            'predicted_revenue': float(f.forecast_revenue),
            'lower_bound':       float(f.lower_bound),
            'upper_bound':       float(f.upper_bound),
        }
        for f in forecasts
    ]

    return JsonResponse({
        'status': 'success',
        'meta': {
            'model':               selected_model,
            'horizon_months':      horizon,
            'mape_pct':            metrics.get('mape'),
            'r2':                  metrics.get('r2'),
            'confidence_interval': '95%',
        },
        'data': data,
    })


# ---------------------------------------------------------------------------
# Real-time Prediction API
# ---------------------------------------------------------------------------

#@login_required
def api_realtime_predict(request):
    """
    Real-time prediction endpoint — runs model.predict() live on every call.
    No DB read/write.  Models are cached in-process after first load.

    POST body (JSON):
    {
        "model":           "XGBoost",          // required
        "product_id":      1,                  // required
        "base_price":      1200.0,             // required
        "lag_1_revenue":   15000.0,            // required
        "lag_2_revenue":   14200.0,            // required
        "start_year":      2025,               // required
        "start_month":     10,                 // required
        "horizon_months":  12                  // optional, default 12
    }

    Response:
    {
        "status":  "success" | "error",
        "model":   str,
        "horizon": int,
        "data": [{"month": "2025-10", "predicted": ..., "lower_bound": ..., "upper_bound": ...}]
    }
    """
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'POST required'}, status=405)

    try:
        body = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({'status': 'error', 'message': 'Invalid JSON body'}, status=400)

    # ── Validate required fields ──────────────────────────────────────────────
    required = ['model', 'product_id', 'base_price', 'lag_1_revenue',
                'lag_2_revenue', 'start_year', 'start_month']
    missing  = [f for f in required if f not in body]
    if missing:
        return JsonResponse({
            'status': 'error',
            'message': f'Missing fields: {", ".join(missing)}'
        }, status=400)

    valid_models = ['XGBoost', 'Prophet', 'ARIMA', 'LSTM']
    model_name   = body.get('model', 'XGBoost')
    if model_name not in valid_models:
        return JsonResponse({
            'status': 'error',
            'message': f'Unknown model. Choose from: {valid_models}'
        }, status=400)

    try:
        product_id      = int(body['product_id'])
        base_price      = float(body['base_price'])
        lag_1_revenue   = float(body['lag_1_revenue'])
        lag_2_revenue   = float(body['lag_2_revenue'])
        start_year      = int(body['start_year'])
        start_month     = int(body['start_month'])
        horizon_months  = int(body.get('horizon_months', 12))
        horizon_months  = max(1, min(horizon_months, 36))  # clamp 1–36
    except (TypeError, ValueError) as exc:
        return JsonResponse({'status': 'error', 'message': f'Type error: {exc}'}, status=400)

    # ── Run live prediction ───────────────────────────────────────────────────
    from analytics.ml.realtime_predict import predict_horizon, _load_metrics

    results = predict_horizon(
        model_name      = model_name,
        product_id      = product_id,
        base_price      = base_price,
        lag_1_revenue   = lag_1_revenue,
        lag_2_revenue   = lag_2_revenue,
        start_year      = start_year,
        start_month     = start_month,
        horizon_months  = horizon_months,
    )

    # Surface any model-load error as a top-level error response
    if results and results[0].get('error'):
        return JsonResponse({
            'status':  'error',
            'message': results[0]['error'],
        }, status=500)

    metrics = _load_metrics().get(model_name, {})

    return JsonResponse({
        'status':  'success',
        'model':   model_name,
        'horizon': horizon_months,
        'meta': {
            'mape': metrics.get('mape'),
            'r2':   metrics.get('r2'),
            'confidence_interval': '95%',
        },
        'data': results,
    })

---------------------
v - 1

import csv
import json
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.db.models import Sum, Count, Avg
from django.db.models.functions import TruncMonth
from django.contrib.auth.decorators import login_required
from .models import Customer, Product, Sales_Transaction, Sales_Forecast, FM_Customer_Segment
from .ml.train_model import load_persisted_metrics

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_feature_importances(model_name):
    """
    Return real feature importances from the saved model file.
    All heavy imports are lazy so Django startup is unaffected.
    """
    import os
    try:
        import joblib
        import numpy as np
    except ImportError:
        return None
    from django.conf import settings

    MODEL_DIR = os.path.join(settings.BASE_DIR, 'analytics', 'ml')
    path = os.path.join(MODEL_DIR, f'{model_name.lower()}_model.pkl')
    if not os.path.exists(path):
        return None

    feature_labels = {
        'product_id':     'Product Identity',
        'year':           'Year Trend',
        'month_num':      'Seasonality (Month)',
        'base_price':     'Base Price',
        'lag_1_revenue':  'Lag Revenue (t-1)',
        'lag_2_revenue':  'Lag Revenue (t-2)',
    }
    feature_order = list(feature_labels.keys())

    try:
        model = joblib.load(path)
        if hasattr(model, 'feature_importances_'):
            fi = model.feature_importances_
            pairs = sorted(
                [(feature_labels[k], round(v * 100, 1)) for k, v in zip(feature_order, fi)],
                key=lambda x: -x[1]
            )
            return pairs[:5]   # top-5 drivers
        elif hasattr(model, 'coef_'):
            # Ridge — use absolute normalised coefficients
            import numpy as np
            coef = model.coef_
            abs_coef = np.abs(coef)
            total = abs_coef.sum() or 1
            pairs = sorted(
                [(feature_labels[k], round(v / total * 100, 1)) for k, v in zip(feature_order, abs_coef)],
                key=lambda x: -x[1]
            )
            return pairs[:5]
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

@login_required
def dashboard_view(request):
    industry_filter = request.GET.get('industry', '')
    region_filter   = request.GET.get('region', '')

    transactions = Sales_Transaction.objects.all()
    if industry_filter:
        transactions = transactions.filter(customer__industry=industry_filter)
    if region_filter:
        transactions = transactions.filter(customer__region=region_filter)

    total_revenue   = transactions.aggregate(Sum('revenue'))['revenue__sum'] or 0
    total_customers = transactions.values('customer').distinct().count()
    total_products  = Product.objects.count()

    industry_revenue = list(
        transactions.values('customer__industry')
        .annotate(revenue=Sum('revenue'))
        .order_by('-revenue')
    )
    chart_labels = [item['customer__industry'] for item in industry_revenue]
    chart_data   = [float(item['revenue']) for item in industry_revenue]

    region_revenue = list(
        transactions.values('customer__region')
        .annotate(revenue=Sum('revenue'))
        .order_by('-revenue')
    )
    region_labels = [item['customer__region'] for item in region_revenue]
    region_data   = [float(item['revenue']) for item in region_revenue]

    context = {
        'total_revenue':   total_revenue,
        'total_customers': total_customers,
        'total_products':  total_products,
        'chart_labels':    json.dumps(chart_labels),
        'chart_data':      json.dumps(chart_data),
        'region_labels':   json.dumps(region_labels),
        'region_data':     json.dumps(region_data),
        'industries':      ['Technology', 'Healthcare', 'Finance', 'Education', 'Manufacturing', 'Retail'],
        'regions':         ['North America', 'Europe', 'Asia Pacific', 'Latin America'],
        'selected_industry': industry_filter,
        'selected_region':   region_filter,
    }
    return render(request, 'analytics/dashboard.html', context)


# ---------------------------------------------------------------------------
# Sales Forecast
# ---------------------------------------------------------------------------

@login_required
def sales_forecast_view(request):
    # ── Historical data (last 12 months) ─────────────────────────────────────
    historical_data = list(
        Sales_Transaction.objects
        .annotate(month=TruncMonth('transaction_date'))
        .values('month')
        .annotate(total_revenue=Sum('revenue'))
        .order_by('month')
    )
    if len(historical_data) > 12:
        historical_data = historical_data[-12:]

    hist_dates    = [d['month'].strftime("%Y-%m") for d in historical_data]
    hist_revenues = [float(d['total_revenue']) for d in historical_data]

    # ── Parameters ───────────────────────────────────────────────────────────
    selected_model = request.GET.get('model', 'XGBoost')
    try:
        horizon_months = int(request.GET.get('horizon', '12'))
    except ValueError:
        horizon_months = 12
    horizon_str = str(horizon_months)

    models_list = ['XGBoost', 'Prophet', 'ARIMA', 'LSTM']
    if selected_model not in models_list:
        selected_model = 'XGBoost'

    # ── Load persisted real metrics ───────────────────────────────────────────
    persisted = load_persisted_metrics()

    # ── Build model comparison table ─────────────────────────────────────────
    model_metrics = []
    best_model    = None
    best_mape     = float('inf')

    for model_name in models_list:
        qs = Sales_Forecast.objects.filter(model_version=model_name)

        if qs.exists():
            # Use persisted training metrics (real values); fall back to DB avg
            pm = persisted.get(model_name, {})
            avg_mape = pm.get('mape', float(qs.aggregate(Avg('mape'))['mape__avg'] or 5.0))
            mae      = pm.get('mae',  0.0)
            rmse     = pm.get('rmse', 0.0)
            r2       = pm.get('r2',   0.0)
        else:
            avg_mape = 0.0
            mae = rmse = r2 = 0.0

        model_metrics.append({
            'name': model_name,
            'mape': round(avg_mape, 2),
            'mae':  round(mae, 2),
            'rmse': round(rmse, 2),
            'r2':   round(r2, 4),
        })

        if avg_mape < best_mape and avg_mape > 0:
            best_mape  = avg_mape
            best_model = model_name

    # ── Selected-model forecast data ─────────────────────────────────────────
    model_qs = Sales_Forecast.objects.filter(model_version=selected_model)

    agg_forecasts = list(
        model_qs.values('forecast_date')
        .annotate(
            total_revenue=Sum('forecast_revenue'),
            total_lower=Sum('lower_bound'),
            total_upper=Sum('upper_bound'),
        )
        .order_by('forecast_date')[:horizon_months]
    )

    forecast_dates   = [f['forecast_date'].strftime("%Y-%m") for f in agg_forecasts]
    forecast_revenues = [float(f['total_revenue']) for f in agg_forecasts]
    forecast_lower    = [float(f['total_lower'])   for f in agg_forecasts]
    forecast_upper    = [float(f['total_upper'])   for f in agg_forecasts]

    # Merge timelines: nulls where one series is absent
    all_dates = hist_dates + forecast_dates
    aligned_hist     = hist_revenues + [None] * len(forecast_dates)
    aligned_forecast = [None] * len(hist_revenues) + forecast_revenues
    aligned_lower    = [None] * len(hist_revenues) + forecast_lower
    aligned_upper    = [None] * len(hist_revenues) + forecast_upper

    # Bridge gap: copy last historical point to first forecast position
    if hist_revenues and forecast_revenues:
        bridge = len(hist_revenues) - 1
        aligned_forecast[bridge] = hist_revenues[-1]
        aligned_lower[bridge]    = hist_revenues[-1]
        aligned_upper[bridge]    = hist_revenues[-1]

    # ── Selected model's metrics ──────────────────────────────────────────────
    selected_metrics = next(
        (m for m in model_metrics if m['name'] == selected_model),
        model_metrics[0] if model_metrics else {}
    )
    avg_mape = selected_metrics.get('mape', 0)
    mae      = selected_metrics.get('mae',  0)
    rmse     = selected_metrics.get('rmse', 0)
    r2       = selected_metrics.get('r2',   0)

    avg_revenue = (sum(forecast_revenues) / len(forecast_revenues)) if forecast_revenues else 0

    # ── Top products over the selected horizon ────────────────────────────────
    unique_forecast_dates = list(
        model_qs.values_list('forecast_date', flat=True)
        .distinct().order_by('forecast_date')[:horizon_months]
    )
    top_products = list(
        model_qs.filter(forecast_date__in=unique_forecast_dates)
        .values('product__product_name')
        .annotate(total_forecast=Sum('forecast_revenue'))
        .order_by('-total_forecast')[:5]
    )

    # ── Real feature importances ──────────────────────────────────────────────
    feature_importances = _get_feature_importances(selected_model)

    context = {
        'selected_model':      selected_model,
        'horizon':             horizon_str,
        'models':              models_list,
        'model_metrics':       model_metrics,
        'best_model':          best_model or selected_model,
        'forecast_dates':      json.dumps(all_dates),
        'historical_revenues': json.dumps(aligned_hist),
        'forecast_revenues':   json.dumps(aligned_forecast),
        'lower_bounds':        json.dumps(aligned_lower),
        'upper_bounds':        json.dumps(aligned_upper),
        'top_products':        top_products,
        'mape':                round(avg_mape, 2),
        'mae':                 round(mae, 2),
        'rmse':                round(rmse, 2),
        'r2':                  round(r2, 4),
        'avg_forecast_revenue': round(avg_revenue, 2),
        'feature_importances': feature_importances,  # list[(label, pct)] or None
    }
    return render(request, 'analytics/sales_forecast.html', context)


# ---------------------------------------------------------------------------
# Customer Segmentation
# ---------------------------------------------------------------------------

@login_required
def segmentation_view(request):
    segments = FM_Customer_Segment.objects.select_related('customer').all()

    segment_counts = list(
        FM_Customer_Segment.objects.values('segment')
        .annotate(count=Count('segment'))
        .order_by('-count')
    )
    labels = [s['segment'] for s in segment_counts]
    data   = [s['count']   for s in segment_counts]

    color_map = {
        'Champions': '#10b981',
        'Loyal':     '#3b82f6',
        'At Risk':   '#f59e0b',
        'New':       '#8b5cf6',
        'Lost':      '#ef4444',
        'Promising': '#06b6d4',
    }

    groups = {}
    for seg in segments:
        s_name = seg.segment
        groups.setdefault(s_name, []).append({
            'x': float(seg.recency),
            'y': float(seg.monetary),
        })

    scatter_datasets = [
        {
            'label':           s_name,
            'data':            points,
            'backgroundColor': color_map.get(s_name, '#06b6d4'),
            'borderColor':     color_map.get(s_name, '#06b6d4'),
            'pointRadius':     5,
            'pointHoverRadius': 8,
        }
        for s_name, points in groups.items()
    ]

    context = {
        'segments':         segments,
        'chart_labels':     json.dumps(labels),
        'chart_data':       json.dumps(data),
        'scatter_datasets': json.dumps(scatter_datasets),
    }
    return render(request, 'analytics/segmentation.html', context)


# ---------------------------------------------------------------------------
# CSV Exports
# ---------------------------------------------------------------------------

@login_required
def export_report_view(request):
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="sales_report.csv"'
    writer = csv.writer(response)
    writer.writerow(['Transaction ID', 'Date', 'Customer', 'Industry', 'Product', 'Quantity', 'Revenue'])
    for txn in Sales_Transaction.objects.select_related('customer', 'product').all():
        writer.writerow([
            txn.transaction_id, txn.transaction_date,
            txn.customer.customer_name, txn.customer.industry,
            txn.product.product_name, txn.quantity, txn.revenue,
        ])
    return response


@login_required
def export_forecast_csv(request):
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="sales_forecast_export.csv"'
    writer = csv.writer(response)
    writer.writerow(['Forecast Date', 'Product', 'Model', 'Predicted Revenue', 'Lower Bound', 'Upper Bound', 'MAPE (%)'])
    for f in Sales_Forecast.objects.select_related('product').order_by('model_version', 'forecast_date'):
        writer.writerow([
            f.forecast_date, f.product.product_name, f.model_version,
            f.forecast_revenue, f.lower_bound, f.upper_bound, f.mape,
        ])
    return response


# ---------------------------------------------------------------------------
# JSON API
# ---------------------------------------------------------------------------

@login_required
def api_forecast_view(request):
    """
    Serves forecast data as JSON for downstream consumption.
    Query params: ?model=XGBoost  (default)  &horizon=12
    """
    selected_model = request.GET.get('model', 'XGBoost')
    try:
        horizon = int(request.GET.get('horizon', '24'))
    except ValueError:
        horizon = 24

    forecasts = (
        Sales_Forecast.objects
        .filter(model_version=selected_model)
        .order_by('forecast_date')[:horizon]
    )

    metrics = load_persisted_metrics().get(selected_model, {})

    data = [
        {
            'date':              f.forecast_date.strftime('%Y-%m-%d'),
            'product':           f.product.product_name,
            'predicted_revenue': float(f.forecast_revenue),
            'lower_bound':       float(f.lower_bound),
            'upper_bound':       float(f.upper_bound),
        }
        for f in forecasts
    ]

    return JsonResponse({
        'status': 'success',
        'meta': {
            'model':               selected_model,
            'horizon_months':      horizon,
            'mape_pct':            metrics.get('mape'),
            'r2':                  metrics.get('r2'),
            'confidence_interval': '95%',
        },
        'data': data,
    })
'''