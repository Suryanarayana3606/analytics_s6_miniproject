# Predictive Sales Analytics

A Django-based sales analytics and forecasting platform with ML-powered revenue predictions, RFM customer segmentation, and an interactive dashboard.

---

## Features

| Feature | Details |
|---|---|
| **Sales Dashboard** | Revenue by industry & region with live filters |
| **Sales Forecast** | 24-month rolling forecasts from 4 models with 95% CI |
| **Customer Segmentation** | RFM scoring, Champions/Loyal/At-Risk/Lost classification |
| **Export** | CSV export for transactions and forecasts |
| **JSON API** | `/api/forecast/?model=XGBoost&horizon=12` |
| **Admin** | Full Django admin with search, filter, date hierarchy |

---

## Quick Start

### 1. Clone and install dependencies

```bash
git clone <repo-url>
cd mini_project
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Apply migrations

```bash
python manage.py migrate
```

### 3. Seed demo data and train ML models

```bash
python populate_db.py
```

This creates 50 customers, 15 products, ~3 years of transactions, computes RFM segments, trains all four models, and generates 24-month forecasts.

### 4. Create a superuser

```bash
python manage.py createsuperuser
```

### 5. Run the development server

```bash
python manage.py runserver
```

Visit `http://127.0.0.1:8000` — log in with your superuser credentials.

---

## Re-training the Models

After adding new transaction data, re-run the ML pipeline:

```bash
python manage.py train_sales_model
```

This re-trains all models, recomputes metrics, and overwrites forecasts in the database.

---

## ML Models

All models use 6 features: `product_id`, `year`, `month_num`, `base_price`, `lag_1_revenue`, `lag_2_revenue`.

| UI Label | sklearn Class | Notes |
|---|---|---|
| XGBoost | `GradientBoostingRegressor` (200 trees) | Fallback — install `xgboost` for the real thing |
| Prophet | `RandomForestRegressor` (200 trees) | Proxy |
| ARIMA | `Ridge` (α=10) | Proxy; α tuned to prevent year-coeff blow-up |
| LSTM | `GradientBoostingRegressor` (150 trees) | Proxy |

**Confidence intervals** are 95% prediction bands: `pred ± 1.96 × residual_std`, where `residual_std` is computed on the held-out test set. Not a fixed ±15%.

**Train/test split** is chronological (last 20% of rows by time) — no data leakage.

Metrics (`MAPE`, `MAE`, `RMSE`, `R²`) are computed on the held-out test set and persisted to `analytics/ml/model_metrics.json`.

---

## Project Structure

```
mini_project/
├── analytics/
│   ├── ml/
│   │   ├── features.py          # Feature engineering pipeline
│   │   ├── train_model.py       # Training + metric persistence
│   │   ├── predict.py           # Rolling forecast generation
│   │   └── model_metrics.json   # Auto-generated after training
│   ├── management/commands/
│   │   └── train_sales_model.py # python manage.py train_sales_model
│   ├── templates/analytics/
│   │   ├── base.html
│   │   ├── dashboard.html
│   │   ├── sales_forecast.html
│   │   └── segmentation.html
│   ├── admin.py
│   ├── models.py
│   ├── tests.py
│   ├── urls.py
│   └── views.py
├── config/
│   ├── settings.py              # Env-var driven; safe defaults for dev
│   ├── urls.py
│   └── wsgi.py
├── populate_db.py               # Demo data + ML pipeline seeder
├── requirements.txt
└── README.md
```

---

## Running Tests

```bash
python manage.py test analytics
```

Covers: model `__str__`, ordering, all views (authenticated + redirect-when-anonymous), CSV exports, JSON API, and ML feature engineering.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `DJANGO_SECRET_KEY` | insecure dev key | Set a strong random key in production |
| `DJANGO_DEBUG` | `True` | Set to `False` in production |
| `DJANGO_ALLOWED_HOSTS` | `localhost 127.0.0.1` | Space-separated list |
| `DJANGO_LOG_LEVEL` | `WARNING` | Django internal log verbosity |

---

## Production Checklist

- [ ] Set `DJANGO_SECRET_KEY` to a randomly generated value
- [ ] Set `DJANGO_DEBUG=False`
- [ ] Set `DJANGO_ALLOWED_HOSTS` to your domain
- [ ] Run `python manage.py collectstatic`
- [ ] Serve with gunicorn + nginx (or similar)
- [ ] Consider PostgreSQL instead of SQLite for concurrency
