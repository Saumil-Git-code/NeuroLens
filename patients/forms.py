from django import forms
from .models import Patient


class PatientForm(forms.ModelForm):
    """Clinical form to register or update patient information."""

    class Meta:
        model = Patient
        fields = ['full_name', 'age', 'gender', 'contact', 'medical_history']
        widgets = {
            'full_name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'e.g. John Doe',
                'required': True,
            }),
            'age': forms.NumberInput(attrs={
                'class': 'form-control',
                'placeholder': 'Age in years (0-120)',
                'min': '0',
                'max': '120',
                'required': True,
            }),
            'gender': forms.Select(attrs={
                'class': 'form-select',
                'required': True,
            }),
            'contact': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Phone or emergency contact',
            }),
            'medical_history': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 4,
                'placeholder': 'Known medical conditions, prior surgeries, symptoms, allergies...',
            }),
        }
