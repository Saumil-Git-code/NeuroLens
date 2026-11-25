from django.urls import path
from . import views

app_name = 'reports'

urlpatterns = [
    path('', views.export_options, name='export_options'),
    path('csv/', views.export_csv, name='export_csv'),
    path('pdf/<int:scan_id>/', views.generate_pdf, name='generate_pdf'),
]
