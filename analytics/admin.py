from django.contrib import admin
from .models import Customer, Product, Sales_Transaction, Sales_Forecast, FM_Customer_Segment


@admin.register(Customer)
class CustomerAdmin(admin.ModelAdmin):
    list_display  = ('customer_id', 'customer_name', 'customer_type', 'industry', 'region', 'account_value')
    search_fields = ('customer_name', 'industry', 'region')
    list_filter   = ('industry', 'region', 'customer_type')
    ordering      = ('customer_name',)


@admin.register(Product)
class ProductAdmin(admin.ModelAdmin):
    list_display  = ('product_id', 'product_name', 'category', 'base_price')
    search_fields = ('product_name', 'category')
    list_filter   = ('category',)
    ordering      = ('product_name',)


@admin.register(Sales_Transaction)
class SalesTransactionAdmin(admin.ModelAdmin):
    list_display   = ('transaction_id', 'transaction_date', 'customer', 'product', 'quantity', 'unit_price', 'revenue')
    list_filter    = ('transaction_date', 'customer__industry', 'product__category')
    search_fields  = ('customer__customer_name', 'product__product_name')
    date_hierarchy = 'transaction_date'
    ordering       = ('-transaction_date',)
    raw_id_fields  = ('customer', 'product')


@admin.register(Sales_Forecast)
class SalesForecastAdmin(admin.ModelAdmin):
    list_display  = ('forecast_id', 'product', 'forecast_date', 'model_version',
                     'forecast_revenue', 'lower_bound', 'upper_bound', 'mape')
    list_filter   = ('model_version', 'forecast_date', 'product__category')
    search_fields = ('product__product_name',)
    ordering      = ('model_version', 'forecast_date')
    date_hierarchy = 'forecast_date'


@admin.register(FM_Customer_Segment)
class FMCustomerSegmentAdmin(admin.ModelAdmin):
    list_display  = ('customer', 'segment', 'recency', 'frequency', 'monetary',
                     'r_score', 'f_score', 'm_score', 'rfm_score')
    list_filter   = ('segment',)
    search_fields = ('customer__customer_name', 'segment')
    ordering      = ('-rfm_score',)
