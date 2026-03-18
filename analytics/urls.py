from django.urls import path
from . import views

app_name = 'analytics'

urlpatterns = [
    path('',                      views.dashboard_view,         name='dashboard'),
    path('sales-forecast/',       views.sales_forecast_view,    name='sales_forecast'),
    path('segmentation/',         views.segmentation_view,      name='segmentation'),
    path('export/',               views.export_report_view,     name='export_report'),
    path('export-forecast/',      views.export_forecast_csv,    name='export_forecast'),
    # Batch (pre-computed) forecast API
    path('api/forecast/',         views.api_forecast_view,      name='api_forecast'),
    # Real-time (live model.predict) API
    path('api/predict/',          views.api_realtime_predict,   name='api_realtime_predict'),
]


'''
v - 1
from django.urls import path
from . import views

app_name = 'analytics'

urlpatterns = [
    path('', views.dashboard_view, name='dashboard'),
    path('sales-forecast/', views.sales_forecast_view, name='sales_forecast'),
    path('segmentation/', views.segmentation_view, name='segmentation'),
    path('export/', views.export_report_view, name='export_report'),
    path('export-forecast/', views.export_forecast_csv, name='export_forecast'),
    path('api/forecast/', views.api_forecast_view, name='api_forecast'),
]
'''