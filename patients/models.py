from django.db import models
from django.contrib.auth.models import User
from django.core.validators import MinValueValidator, MaxValueValidator


class Patient(models.Model):
    """Patient clinical record with auto-generated code."""
    GENDER_CHOICES = [
        ('M', 'Male'),
        ('F', 'Female'),
        ('O', 'Other'),
    ]
    patient_code = models.CharField(max_length=10, unique=True, editable=False)
    full_name = models.CharField(max_length=200)
    age = models.PositiveIntegerField(validators=[MinValueValidator(0), MaxValueValidator(120)])
    gender = models.CharField(max_length=1, choices=GENDER_CHOICES)
    contact = models.CharField(max_length=100, blank=True)
    medical_history = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    created_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        related_name='created_patients'
    )

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.patient_code} - {self.full_name}"

    def save(self, *args, **kwargs):
        if not self.patient_code:
            last = Patient.objects.order_by('-id').first()
            num = (last.id + 1) if (last and last.id) else 1
            # Ensure unique code if gaps or deletions exist
            code = f"NL-{num:04d}"
            while Patient.objects.filter(patient_code=code).exists():
                num += 1
                code = f"NL-{num:04d}"
            self.patient_code = code
        super().save(*args, **kwargs)


class CareTeam(models.Model):
    """Many-to-many relationship between Doctor and Patient with assignment date."""
    doctor = models.ForeignKey(User, on_delete=models.CASCADE, related_name='care_assignments')
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE, related_name='care_team')
    assigned_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ['doctor', 'patient']
        ordering = ['-assigned_at']

    def __str__(self):
        doctor_name = self.doctor.get_full_name() or self.doctor.username
        return f"Dr. {doctor_name} -> {self.patient.full_name}"
