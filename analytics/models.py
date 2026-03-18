from django.db import models


class Customer(models.Model):
    customer_id   = models.AutoField(primary_key=True)
    customer_name = models.CharField(max_length=255)
    customer_type = models.CharField(max_length=100)
    industry      = models.CharField(max_length=100)
    region        = models.CharField(max_length=100)
    account_value = models.DecimalField(max_digits=15, decimal_places=2)

    class Meta:
        ordering = ['customer_name']

    def __str__(self):
        return f"{self.customer_name} ({self.customer_id})"


class Product(models.Model):
    product_id   = models.AutoField(primary_key=True)
    product_name = models.CharField(max_length=255)
    category     = models.CharField(max_length=100)
    base_price   = models.DecimalField(max_digits=10, decimal_places=2)

    class Meta:
        ordering = ['product_name']

    def __str__(self):
        return self.product_name


class Sales_Transaction(models.Model):
    transaction_id   = models.AutoField(primary_key=True)
    transaction_date = models.DateField(db_index=True)
    customer  = models.ForeignKey(Customer, on_delete=models.CASCADE, db_column='customer_id')
    product   = models.ForeignKey(Product,  on_delete=models.CASCADE, db_column='product_id')
    quantity  = models.IntegerField()
    unit_price = models.DecimalField(max_digits=10, decimal_places=2)
    revenue   = models.DecimalField(max_digits=15, decimal_places=2)

    class Meta:
        ordering = ['-transaction_date']

    def __str__(self):
        return f"TXN-{self.transaction_id} on {self.transaction_date}"


class Sales_Forecast(models.Model):
    forecast_id      = models.AutoField(primary_key=True)
    product          = models.ForeignKey(Product, on_delete=models.CASCADE, db_column='product_id')
    forecast_date    = models.DateField(db_index=True)
    forecast_revenue = models.DecimalField(max_digits=15, decimal_places=2)
    lower_bound      = models.DecimalField(max_digits=15, decimal_places=2)
    upper_bound      = models.DecimalField(max_digits=15, decimal_places=2)
    model_version    = models.CharField(max_length=50, db_index=True)
    mape             = models.DecimalField(
        max_digits=5, decimal_places=2,
        help_text="Mean Absolute Percentage Error (%) on held-out test set"
    )

    class Meta:
        ordering = ['model_version', 'forecast_date']

    def __str__(self):
        return f"Forecast for {self.product.product_name} on {self.forecast_date} [{self.model_version}]"


class FM_Customer_Segment(models.Model):
    segment_id = models.AutoField(primary_key=True)
    customer   = models.OneToOneField(Customer, on_delete=models.CASCADE, db_column='customer_id')
    recency    = models.IntegerField()
    frequency  = models.IntegerField()
    monetary   = models.DecimalField(max_digits=15, decimal_places=2)
    r_score    = models.IntegerField()
    f_score    = models.IntegerField()
    m_score    = models.IntegerField()
    rfm_score  = models.IntegerField()
    segment    = models.CharField(max_length=100, db_index=True)

    class Meta:
        ordering = ['-rfm_score']

    def __str__(self):
        return f"{self.customer.customer_name} → {self.segment} (RFM={self.rfm_score})"
