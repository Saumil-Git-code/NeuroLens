from django.contrib import admin
from .models import MRIScan, Prediction, DoctorReview


@admin.register(MRIScan)
class MRIScanAdmin(admin.ModelAdmin):
    list_display = ['id', 'patient', 'uploaded_by', 'uploaded_at']
    list_filter = ['uploaded_at']
    search_fields = ['patient__patient_code', 'patient__full_name']


@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = ['scan', 'predicted_label', 'confidence', 'status', 'is_mock', 'created_at']
    list_filter = ['status', 'predicted_label', 'is_mock']
    search_fields = ['scan__patient__patient_code', 'predicted_label']


@admin.register(DoctorReview)
class DoctorReviewAdmin(admin.ModelAdmin):
    list_display = ['prediction', 'reviewer', 'final_label', 'agrees_with_ai', 'reviewed_at']
    list_filter = ['agrees_with_ai', 'reviewed_at']
    search_fields = ['prediction__scan__patient__patient_code', 'reviewer__username', 'final_label']
