import os
from django import forms
from PIL import Image
from .models import MRIScan, Prediction, DoctorReview
from patients.models import Patient
from ml import CLASS_NAMES

ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16 MB


class ScanUploadForm(forms.ModelForm):
    """Form for uploading brain MRI scans with validation."""

    class Meta:
        model = MRIScan
        fields = ['patient', 'image', 'notes']
        widgets = {
            'patient': forms.Select(attrs={
                'class': 'form-select',
                'required': True,
            }),
            'image': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': '.jpg,.jpeg,.png,.bmp',
                'required': True,
                'id': 'scanImageInput',
            }),
            'notes': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 3,
                'placeholder': 'Clinical symptoms, scan orientation, sequence notes (e.g. Axial T1-weighted contrast-enhanced)...',
            }),
        }

    def __init__(self, *args, **kwargs):
        user = kwargs.pop('user', None)
        super().__init__(*args, **kwargs)
        if user:
            is_admin = (hasattr(user, 'profile') and user.profile.is_admin) or user.is_superuser
            if not is_admin:
                self.fields['patient'].queryset = Patient.objects.filter(care_team__doctor=user)

    def clean_image(self):
        image = self.cleaned_data.get('image')
        if not image:
            raise forms.ValidationError('An MRI image file is required.')

        # 1. Validate file extension
        ext = os.path.splitext(image.name)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise forms.ValidationError(
                f'Unsupported file format "{ext}". Please upload a JPG, JPEG, PNG, or BMP image.'
            )

        # 2. Validate file size
        if image.size > MAX_FILE_SIZE:
            size_mb = image.size / (1024 * 1024)
            raise forms.ValidationError(
                f'Image file is too large ({size_mb:.1f}MB). Maximum allowed size is 16MB.'
            )

        # 3. Verify real image using Pillow
        try:
            image_copy = image.file
            img = Image.open(image_copy)
            img.verify()
            # Reset file pointer after verify()
            image_copy.seek(0)
        except Exception as exc:
            raise forms.ValidationError('Invalid image file. The uploaded file is corrupted or not a valid image.') from exc

        return image


class ReviewForm(forms.Form):
    """Doctor's clinical review and override form."""
    CLASS_CHOICES = [(c, c.capitalize()) for c in CLASS_NAMES]

    final_label = forms.ChoiceField(
        choices=CLASS_CHOICES,
        widget=forms.Select(attrs={'class': 'form-select'}),
        label='Clinical Diagnosis (Final Label)'
    )
    agrees_with_ai = forms.BooleanField(
        required=False,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input', 'id': 'agreesCheck'}),
        label='I concur with the AI prediction'
    )
    clinical_notes = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 4,
            'placeholder': 'Enter clinical diagnosis rationale, differential diagnosis, recommended next steps...',
        }),
        label='Clinical & Review Notes'
    )
