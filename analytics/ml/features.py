import os
from django.conf import settings


def get_historical_data():
    """
    Aggregate sales revenue by (product, month) from Sales_Transaction.
    Returns an empty DataFrame (with correct columns) if no data exists.

    pandas is imported lazily — Django startup never touches it.
    """
    import pandas as pd
    from django.db.models import Sum
    from django.db.models.functions import TruncMonth
    from analytics.models import Sales_Transaction

    historical_data = list(
        Sales_Transaction.objects
        .annotate(month=TruncMonth('transaction_date'))
        .values('month', 'product_id')
        .annotate(total_revenue=Sum('revenue'))
        .order_by('product_id', 'month')
    )

    if not historical_data:
        return pd.DataFrame(columns=['month', 'product_id', 'total_revenue'])

    df = pd.DataFrame(historical_data)
    df['month']         = pd.to_datetime(df['month'])
    df['total_revenue'] = df['total_revenue'].astype(float)
    return df


def prepare_features(df):
    """
    Engineer calendar + price + lag features for the ML models.
    pandas is imported lazily.
    """
    import pandas as pd
    from analytics.models import Product

    if df.empty:
        return df

    df = df.copy()
    df['year']      = df['month'].dt.year
    df['month_num'] = df['month'].dt.month

    products_qs = list(Product.objects.all().values('product_id', 'base_price'))
    if products_qs:
        products_df = pd.DataFrame(products_qs)
        products_df['base_price'] = products_df['base_price'].astype(float)
        df = df.merge(products_df, on='product_id', how='left')
        df['base_price'] = df['base_price'].fillna(0.0)
    else:
        df['base_price'] = 0.0

    df = df.sort_values(['product_id', 'month']).reset_index(drop=True)

    df['lag_1_revenue'] = (
        df.groupby('product_id')['total_revenue'].shift(1).fillna(0.0)
    )
    df['lag_2_revenue'] = (
        df.groupby('product_id')['total_revenue'].shift(2).fillna(0.0)
    )

    return df


def fetch_and_prepare_data():
    """Full pipeline: fetch → feature-engineer. Entry point for train & predict."""
    return prepare_features(get_historical_data())
