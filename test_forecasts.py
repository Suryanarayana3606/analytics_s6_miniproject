import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from analytics.models import Sales_Forecast, Product

product = Product.objects.first()
forecast_date = Sales_Forecast.objects.filter(product=product).first().forecast_date

print(f"--- Predictions for {product.product_name} on {forecast_date} ---")
forecasts = Sales_Forecast.objects.filter(product=product, forecast_date=forecast_date).order_by('model_version')

for f in forecasts:
    print(f"Model: {f.model_version:10} | Forecast Revenue: {f.forecast_revenue:10.2f} | MAPE: {f.mape:6.2f}")

print("\n--- First 3 predictions for XGBoost ---")
for f in Sales_Forecast.objects.filter(product=product, model_version='XGBoost').order_by('forecast_date')[:3]:
    print(f"{f.forecast_date}: {f.forecast_revenue:.2f}")

print("\n--- First 3 predictions for Prophet ---")
for f in Sales_Forecast.objects.filter(product=product, model_version='Prophet').order_by('forecast_date')[:3]:
    print(f"{f.forecast_date}: {f.forecast_revenue:.2f}")

print("\n--- First 3 predictions for ARIMA ---")
for f in Sales_Forecast.objects.filter(product=product, model_version='ARIMA').order_by('forecast_date')[:3]:
    print(f"{f.forecast_date}: {f.forecast_revenue:.2f}")

print("\n--- First 3 predictions for LSTM ---")
for f in Sales_Forecast.objects.filter(product=product, model_version='LSTM').order_by('forecast_date')[:3]:
    print(f"{f.forecast_date}: {f.forecast_revenue:.2f}")
