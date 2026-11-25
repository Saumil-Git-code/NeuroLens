from django.urls import path
from . import views

app_name = 'scans'

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('upload/', views.scan_upload, name='upload'),
    path('result/<int:pk>/', views.scan_result, name='result'),
    path('review-queue/', views.review_queue, name='review_queue'),
    path('review/<int:pk>/', views.review_scan, name='review'),
    path('api/search/', views.search_api, name='search_api'),
]
