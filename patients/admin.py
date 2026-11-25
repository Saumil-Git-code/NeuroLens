from django.contrib import admin
from .models import Patient, CareTeam


@admin.register(Patient)
class PatientAdmin(admin.ModelAdmin):
    list_display = ['patient_code', 'full_name', 'age', 'gender', 'created_at', 'created_by']
    list_filter = ['gender', 'created_at']
    search_fields = ['patient_code', 'full_name', 'contact']
    readonly_fields = ['patient_code', 'created_at']


@admin.register(CareTeam)
class CareTeamAdmin(admin.ModelAdmin):
    list_display = ['doctor', 'patient', 'assigned_at']
    list_filter = ['assigned_at']
    search_fields = ['doctor__username', 'doctor__first_name', 'doctor__last_name', 'patient__full_name', 'patient__patient_code']
