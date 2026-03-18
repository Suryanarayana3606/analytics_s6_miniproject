"""
Test suite for the analytics app.
Run with:  python manage.py test analytics
"""
from decimal import Decimal
from datetime import date

from django.test import TestCase, Client
from django.contrib.auth.models import User
from django.urls import reverse

from analytics.models import (
    Customer, Product, Sales_Transaction,
    Sales_Forecast, FM_Customer_Segment,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_product(name='Widget', price=100):
    return Product.objects.create(product_name=name, category='Software', base_price=Decimal(price))


def _make_customer(name='Acme'):
    return Customer.objects.create(
        customer_name=name, customer_type='Enterprise',
        industry='Technology', region='North America',
        account_value=Decimal('50000'),
    )


def _make_txn(customer, product, rev=500, d=None):
    d = d or date(2024, 1, 15)
    return Sales_Transaction.objects.create(
        transaction_date=d, customer=customer, product=product,
        quantity=5, unit_price=Decimal('100'), revenue=Decimal(rev),
    )


def _make_forecast(product, model='XGBoost', d=None, rev=1000):
    d = d or date(2025, 1, 1)
    return Sales_Forecast.objects.create(
        product=product, forecast_date=d,
        forecast_revenue=Decimal(rev),
        lower_bound=Decimal(rev * 0.85),
        upper_bound=Decimal(rev * 1.15),
        model_version=model,
        mape=Decimal('5.00'),
    )


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

class CustomerModelTest(TestCase):
    def test_str(self):
        c = _make_customer('Globex')
        self.assertIn('Globex', str(c))

    def test_ordering(self):
        _make_customer('Zeta')
        _make_customer('Alpha')
        names = list(Customer.objects.values_list('customer_name', flat=True))
        self.assertEqual(names, sorted(names))


class ProductModelTest(TestCase):
    def test_str(self):
        p = _make_product('Gadget')
        self.assertEqual(str(p), 'Gadget')

    def test_ordering(self):
        _make_product('Zebra')
        _make_product('Apple')
        names = list(Product.objects.values_list('product_name', flat=True))
        self.assertEqual(names, sorted(names))


class SalesForecastModelTest(TestCase):
    def test_str_contains_model_version(self):
        p = _make_product()
        f = _make_forecast(p, model='ARIMA')
        self.assertIn('ARIMA', str(f))

    def test_non_negative_bounds(self):
        p = _make_product()
        f = _make_forecast(p, rev=500)
        self.assertGreaterEqual(f.lower_bound, 0)
        self.assertLessEqual(f.lower_bound, f.forecast_revenue)
        self.assertGreaterEqual(f.upper_bound, f.forecast_revenue)


class FMCustomerSegmentModelTest(TestCase):
    def test_str_contains_segment(self):
        c = _make_customer()
        seg = FM_Customer_Segment.objects.create(
            customer=c, recency=10, frequency=20,
            monetary=Decimal('5000'), r_score=5, f_score=5, m_score=5,
            rfm_score=555, segment='Champions',
        )
        self.assertIn('Champions', str(seg))


# ---------------------------------------------------------------------------
# View tests — authenticated
# ---------------------------------------------------------------------------

class AuthenticatedViewTest(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user('testuser', password='testpass123')
        self.client.login(username='testuser', password='testpass123')
        self.product  = _make_product()
        self.customer = _make_customer()
        _make_txn(self.customer, self.product)
        _make_forecast(self.product)

    def test_dashboard_200(self):
        resp = self.client.get(reverse('analytics:dashboard'))
        self.assertEqual(resp.status_code, 200)

    def test_dashboard_filter_by_industry(self):
        resp = self.client.get(reverse('analytics:dashboard'), {'industry': 'Technology'})
        self.assertEqual(resp.status_code, 200)

    def test_dashboard_context_keys(self):
        resp = self.client.get(reverse('analytics:dashboard'))
        for key in ('total_revenue', 'total_customers', 'total_products', 'chart_labels', 'chart_data'):
            self.assertIn(key, resp.context)

    def test_sales_forecast_200(self):
        resp = self.client.get(reverse('analytics:sales_forecast'))
        self.assertEqual(resp.status_code, 200)

    def test_sales_forecast_model_param(self):
        for model in ('XGBoost', 'Prophet', 'ARIMA', 'LSTM'):
            resp = self.client.get(reverse('analytics:sales_forecast'), {'model': model})
            self.assertEqual(resp.status_code, 200)

    def test_sales_forecast_context_keys(self):
        resp = self.client.get(reverse('analytics:sales_forecast'))
        for key in ('selected_model', 'model_metrics', 'best_model',
                    'forecast_dates', 'mape', 'mae', 'rmse', 'r2'):
            self.assertIn(key, resp.context)

    def test_segmentation_200(self):
        resp = self.client.get(reverse('analytics:segmentation'))
        self.assertEqual(resp.status_code, 200)

    def test_export_report_csv(self):
        resp = self.client.get(reverse('analytics:export_report'))
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp['Content-Type'], 'text/csv')

    def test_export_forecast_csv(self):
        resp = self.client.get(reverse('analytics:export_forecast'))
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp['Content-Type'], 'text/csv')

    def test_api_forecast_json(self):
        resp = self.client.get(reverse('analytics:api_forecast'))
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data['status'], 'success')
        self.assertIn('meta', data)
        self.assertIn('data', data)

    def test_api_forecast_model_filter(self):
        _make_forecast(self.product, model='Prophet')
        resp = self.client.get(reverse('analytics:api_forecast'), {'model': 'Prophet'})
        self.assertEqual(resp.json()['meta']['model'], 'Prophet')


class UnauthenticatedRedirectTest(TestCase):
    """All @login_required views must redirect to login, not 200 or 500."""
    PROTECTED_URLS = [
        'analytics:dashboard',
        'analytics:sales_forecast',
        'analytics:segmentation',
        'analytics:export_report',
        'analytics:export_forecast',
        'analytics:api_forecast',
    ]

    def test_redirect_when_anonymous(self):
        for url_name in self.PROTECTED_URLS:
            resp = self.client.get(reverse(url_name))
            self.assertIn(resp.status_code, (302, 301),
                          msg=f"{url_name} did not redirect anonymous user")


# ---------------------------------------------------------------------------
# ML feature engineering tests (no DB needed)
# ---------------------------------------------------------------------------

class FeatureEngineeringTest(TestCase):
    def test_empty_dataframe_returns_empty(self):
        import pandas as pd
        from analytics.ml.features import prepare_features
        empty = pd.DataFrame(columns=['month', 'product_id', 'total_revenue'])
        result = prepare_features(empty)
        self.assertTrue(result.empty)

    def test_lag_features_computed(self):
        import pandas as pd
        import numpy as np
        from analytics.ml.features import prepare_features

        p = _make_product('TestProd', price=200)

        df = pd.DataFrame([
            {'month': pd.Timestamp('2024-01-01'), 'product_id': p.product_id, 'total_revenue': 1000.0},
            {'month': pd.Timestamp('2024-02-01'), 'product_id': p.product_id, 'total_revenue': 1100.0},
            {'month': pd.Timestamp('2024-03-01'), 'product_id': p.product_id, 'total_revenue': 1200.0},
        ])
        result = prepare_features(df)

        # Row 0: both lags are 0 (no prior data)
        self.assertEqual(result.iloc[0]['lag_1_revenue'], 0.0)
        self.assertEqual(result.iloc[0]['lag_2_revenue'], 0.0)
        # Row 1: lag_1 = Jan revenue
        self.assertAlmostEqual(result.iloc[1]['lag_1_revenue'], 1000.0)
        # Row 2: lag_1 = Feb, lag_2 = Jan
        self.assertAlmostEqual(result.iloc[2]['lag_1_revenue'], 1100.0)
        self.assertAlmostEqual(result.iloc[2]['lag_2_revenue'], 1000.0)
