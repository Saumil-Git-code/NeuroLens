from django.urls import path
from . import views

app_name = 'patients'

urlpatterns = [
    path('', views.patient_list, name='list'),
    path('create/', views.patient_create, name='create'),
    path('detail/<int:pk>/', views.patient_detail, name='detail'),
    path('edit/<int:pk>/', views.patient_edit, name='edit'),
    path('delete/<int:pk>/', views.patient_delete, name='delete'),
]
