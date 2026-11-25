from django.db import models
from django.contrib.auth.models import User
from patients.models import Patient


class MRIScan(models.Model):
    """Brain MRI scan image linked to a patient."""
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE, related_name='scans')
    image = models.ImageField(upload_to='scans/%Y/%m/')
    uploaded_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    notes = models.TextField(blank=True)

    class Meta:
        ordering = ['-uploaded_at']

    def __str__(self):
        return f"Scan #{self.id} - {self.patient.patient_code}"


class Prediction(models.Model):
    """AI prediction result for an MRI scan."""
    STATUS_CHOICES = [
        ('AUTO_ACCEPTED', 'Auto Accepted'),
        ('NEEDS_REVIEW', 'Needs Review'),
        ('REVIEWED', 'Reviewed'),
    ]
    scan = models.OneToOneField(MRIScan, on_delete=models.CASCADE, related_name='prediction')
    predicted_label = models.CharField(max_length=20)
    confidence = models.FloatField()
    probabilities = models.JSONField(default=dict, help_text='Probabilities for all 4 classes')
    heatmap_image = models.ImageField(upload_to='heatmaps/', null=True, blank=True)
    model_name = models.CharField(max_length=50, default='vgg16')
    is_mock = models.BooleanField(default=False)
    status = models.CharField(max_length=15, choices=STATUS_CHOICES, default='NEEDS_REVIEW')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Prediction: {self.predicted_label} ({self.confidence:.1%})"


class DoctorReview(models.Model):
    """Doctor's review of an AI prediction (human-in-the-loop)."""
    prediction = models.OneToOneField(Prediction, on_delete=models.CASCADE, related_name='review')
    reviewer = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    final_label = models.CharField(max_length=20)
    agrees_with_ai = models.BooleanField()
    clinical_notes = models.TextField(blank=True)
    reviewed_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-reviewed_at']

    def __str__(self):
        if self.reviewer:
            reviewer_name = self.reviewer.get_full_name() or self.reviewer.username
        else:
            reviewer_name = 'Unknown Doctor'
        return f"Review by Dr. {reviewer_name} ({self.final_label})"
