#!/usr/bin/env bash
set -o errexit

# Install dependencies
pip install -r requirements.txt

# Collect static files
python manage.py collectstatic --no-input

# Run migrations
python manage.py migrate

# Seed DB + train models only if DB is empty (first deploy)
python manage.py shell -c "
from analytics.models import Sales_Transaction
if not Sales_Transaction.objects.exists():
    print('Empty DB — running populate_db and training models...')
    import subprocess
    subprocess.run(['python', 'populate_db.py'], check=True)
    subprocess.run(['python', 'manage.py', 'train_sales_model'], check=True)
    print('Setup complete.')
else:
    print('DB already populated — skipping seed.')
"
