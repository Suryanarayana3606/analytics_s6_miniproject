"""
Populate the database with realistic demo data, then run the ML pipeline.

Usage:
    python populate_db.py

This script:
  1. Creates 15 products, 50 customers, ~3 years of sales transactions
  2. Computes RFM segmentation for every customer
  3. Trains all four ML models and generates 24-month forecasts

Run once after migrations:
    python manage.py migrate
    python populate_db.py
"""

import os
import sys
import django
import random
import math
from datetime import date, timedelta
from decimal import Decimal

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from faker import Faker
from django.db import transaction
from django.db.models import Sum

from analytics.models import (
    Customer, Product, Sales_Transaction,
    Sales_Forecast, FM_Customer_Segment,
)

fake = Faker()
random.seed(42)
Faker.seed(42)

CATEGORIES  = ['Software', 'Hardware', 'Services', 'Consulting']
INDUSTRIES  = ['Technology', 'Healthcare', 'Finance', 'Education', 'Manufacturing', 'Retail']
REGIONS     = ['North America', 'Europe', 'Asia Pacific', 'Latin America']
CUST_TYPES  = ['Enterprise', 'Mid-Market', 'SMB']

END_DATE   = date(2025, 9, 30)   # last transaction date
START_DATE = END_DATE - timedelta(days=3 * 365)


@transaction.atomic
def populate_db():
    print("Clearing existing data...")
    Customer.objects.all().delete()
    Product.objects.all().delete()
    # Cascades handle transactions, forecasts, segments

    # ── 1. Products ──────────────────────────────────────────────────────────
    print("Creating 15 products...")
    products = []
    for _ in range(15):
        p = Product.objects.create(
            product_name=fake.company() + ' ' + fake.word().capitalize() + ' Suite',
            category=random.choice(CATEGORIES),
            base_price=Decimal(random.randint(300, 5000)),
        )
        products.append(p)

    # ── 2. Customers ─────────────────────────────────────────────────────────
    print("Creating 50 customers...")
    customers = []
    for _ in range(50):
        c = Customer.objects.create(
            customer_name=fake.company(),
            customer_type=random.choice(CUST_TYPES),
            industry=random.choice(INDUSTRIES),
            region=random.choice(REGIONS),
            account_value=Decimal(random.randint(10_000, 500_000)),
        )
        customers.append(c)

    # ── 3. Sales Transactions (3 years, realistic patterns) ──────────────────
    print("Creating sales transactions (~3 years)...")

    # Each product gets a unique seasonality + growth profile
    profiles = {}
    for p in products:
        profiles[p.product_id] = {
            'base':   float(p.base_price) * random.uniform(8.0, 15.0),
            'shift':  random.uniform(0, 2 * math.pi),
            'growth': random.uniform(0.0003, 0.0015),
        }

    # Pareto customer weights (top 20 % drive most transactions)
    weights = [1.0 / (i + 1) ** 1.5 for i in range(len(customers))]
    total_w = sum(weights)
    weights = [w / total_w for w in weights]

    txns_to_create = []
    current = START_DATE
    while current <= END_DATE:
        day_idx = (current - START_DATE).days
        for p in products:
            prof = profiles[p.product_id]
            seasonal = 1 + 0.25 * math.sin(2 * math.pi * current.month / 12 + prof['shift'])
            trend    = 1 + prof['growth'] * day_idx
            daily_prob = 0.6 * seasonal * trend  # probability of a sale today

            if random.random() < daily_prob:
                cust = random.choices(customers, weights=weights, k=1)[0]
                qty  = random.randint(1, 10)
                unit = float(p.base_price) * random.uniform(0.9, 1.1)
                txns_to_create.append(Sales_Transaction(
                    transaction_date=current,
                    customer=cust,
                    product=p,
                    quantity=qty,
                    unit_price=Decimal(round(unit, 2)),
                    revenue=Decimal(round(unit * qty, 2)),
                ))
        current += timedelta(days=1)

    Sales_Transaction.objects.bulk_create(txns_to_create, batch_size=1000)
    print(f"  Created {len(txns_to_create)} transactions.")

    # ── 4. RFM Segmentation ──────────────────────────────────────────────────
    print("Computing RFM segmentation...")
    rfm_rows = []
    for c in customers:
        txns = Sales_Transaction.objects.filter(customer=c)
        if not txns.exists():
            continue
        latest      = txns.order_by('-transaction_date').first().transaction_date
        recency     = (END_DATE - latest).days
        frequency   = txns.count()
        monetary    = float(txns.aggregate(Sum('revenue'))['revenue__sum'] or 0)
        rfm_rows.append({'customer': c, 'recency': recency,
                         'frequency': frequency, 'monetary': monetary})

    if rfm_rows:
        import pandas as pd
        df = pd.DataFrame(rfm_rows)
        for col, asc in [('recency', False), ('frequency', True), ('monetary', True)]:
            rank_col   = col + '_rank'
            score_col  = col[0] + '_score' if col != 'frequency' else 'f_score'
            df[rank_col]  = df[col].rank(method='first', ascending=asc)
            df[score_col] = pd.qcut(df[rank_col], 5, labels=[1, 2, 3, 4, 5]).astype(int)

        segs_to_create = []
        for _, row in df.iterrows():
            r, f, m = int(row['r_score']), int(row['f_score']), int(row['m_score'])
            if   r >= 4 and f >= 4 and m >= 4: seg = 'Champions'
            elif r >= 3 and f >= 3 and m >= 3: seg = 'Loyal'
            elif r <= 2 and m >= 3:             seg = 'At Risk'
            elif r >= 4 and f <= 2:             seg = 'New'
            elif r <= 2 and f <= 2:             seg = 'Lost'
            else:                               seg = 'Promising'

            segs_to_create.append(FM_Customer_Segment(
                customer=row['customer'],
                recency=int(row['recency']), frequency=int(row['frequency']),
                monetary=Decimal(round(row['monetary'], 2)),
                r_score=r, f_score=f, m_score=m,
                rfm_score=int(f'{r}{f}{m}'),
                segment=seg,
            ))
        FM_Customer_Segment.objects.bulk_create(segs_to_create)
        print(f"  Created {len(segs_to_create)} RFM segments.")

    print("\nDatabase population complete.")


def run_ml_pipeline():
    """Train models and generate forecasts using the real ML pipeline."""
    from analytics.ml.train_model import train_and_save_model
    from analytics.ml.predict import run_predictions

    print("\nStarting ML pipeline...")
    metrics = train_and_save_model()
    if not metrics:
        print("ERROR: Training failed.")
        return

    print("\n  Model performance on held-out test set:")
    for name, m in metrics.items():
        print(f"    {name:10s}  MAPE={m['mape']:.2f}%  RMSE={m['rmse']:,.0f}  R²={m['r2']:.4f}")

    run_predictions(metrics)
    print("\nML pipeline complete. 24-month forecasts saved.")


if __name__ == '__main__':
    populate_db()
    run_ml_pipeline()
