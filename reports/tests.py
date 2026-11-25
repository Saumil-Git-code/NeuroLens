"""
Tests for NeuroLens Reports Application.
========================================
Covers CSV clinical data export with role-based row filtering,
PDF clinical diagnostic report generation with ReportLab, and
access authorization controls for patient reports.
"""

import io
import shutil
import tempfile
from PIL import Image as PILImage

from django.test import TestCase, override_settings
from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse
from django.contrib.messages import get_messages

from accounts.models import Profile
from patients.models import Patient, CareTeam
from scans.models import MRIScan, Prediction, DoctorReview


def create_test_image(filename='test.jpg', size=(224, 224), format='JPEG'):
    """Create a simple test image in memory."""
    img = PILImage.new('RGB', size, color='gray')
    buffer = io.BytesIO()
    img.save(buffer, format=format)
    buffer.seek(0)
    return SimpleUploadedFile(filename, buffer.read(), content_type=f'image/{format.lower()}')


TEMP_MEDIA_DIR = tempfile.mkdtemp()


@override_settings(MEDIA_ROOT=TEMP_MEDIA_DIR)
class ReportsViewsTests(TestCase):
    """Integration and authorization tests for CSV export and PDF report generation."""

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        shutil.rmtree(TEMP_MEDIA_DIR, ignore_errors=True)

    def setUp(self):
        # Admin user
        self.admin_user = User.objects.create_superuser(
            username='admin_reports',
            password='adminpassword',
            first_name='Admin',
            last_name='User'
        )
        # Doctor user
        self.doctor_user = User.objects.create_user(
            username='dr_reports',
            password='doctorpassword',
            first_name='Leonard',
            last_name='McCoy'
        )
        # Patients
        self.assigned_patient = Patient.objects.create(
            full_name='James Kirk',
            age=34,
            gender='M'
        )
        self.unassigned_patient = Patient.objects.create(
            full_name='Spock',
            age=35,
            gender='M'
        )
        # CareTeam
        CareTeam.objects.create(doctor=self.doctor_user, patient=self.assigned_patient)

        # Scans
        self.assigned_scan = MRIScan.objects.create(
            patient=self.assigned_patient,
            image=create_test_image('kirk_scan.jpg'),
            uploaded_by=self.doctor_user,
            notes='Patient reports headache'
        )
        self.unassigned_scan = MRIScan.objects.create(
            patient=self.unassigned_patient,
            image=create_test_image('spock_scan.jpg'),
            uploaded_by=self.admin_user,
            notes='Routine baseline'
        )

        # Predictions
        self.assigned_pred = Prediction.objects.create(
            scan=self.assigned_scan,
            predicted_label='glioma',
            confidence=0.89,
            probabilities={'glioma': 0.89, 'meningioma': 0.04, 'notumor': 0.03, 'pituitary': 0.04},
            status='AUTO_ACCEPTED',
            is_mock=False
        )
        self.unassigned_pred = Prediction.objects.create(
            scan=self.unassigned_scan,
            predicted_label='notumor',
            confidence=0.96,
            probabilities={'glioma': 0.01, 'meningioma': 0.01, 'notumor': 0.96, 'pituitary': 0.02},
            status='AUTO_ACCEPTED',
            is_mock=False
        )

    def test_export_options_view_returns_200(self):
        """Export options dashboard should render successfully for authenticated users."""
        self.client.login(username='dr_reports', password='doctorpassword')
        response = self.client.get(reverse('reports:export_options'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'reports/export_options.html')

    def test_csv_export_returns_200_and_csv_content_type(self):
        """CSV export endpoint must return HTTP 200 and 'text/csv' content type."""
        self.client.login(username='dr_reports', password='doctorpassword')
        response = self.client.get(reverse('reports:export_csv'))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['Content-Type'], 'text/csv')
        self.assertIn('attachment; filename="neurolens_scans_export_', response['Content-Disposition'])

    def test_csv_export_contains_standard_header_row(self):
        """CSV output must contain expected diagnostic and clinical column headers."""
        self.client.login(username='dr_reports', password='doctorpassword')
        response = self.client.get(reverse('reports:export_csv'))
        csv_content = response.content.decode('utf-8')
        first_line = csv_content.splitlines()[0]

        expected_headers = [
            'Scan ID',
            'Patient Code',
            'Patient Name',
            'Age',
            'Gender',
            'Upload Date',
            'Uploaded By',
            'Predicted Label',
            'Confidence Score',
            'AI Status',
            'AI Model',
            'Is Mock Prediction',
            'Doctor Review Done',
            'Doctor Final Label',
            'Agrees With AI',
            'Clinical Notes',
        ]
        for header in expected_headers:
            self.assertIn(header, first_line)

    def test_csv_export_respects_role_filtering_for_doctor(self):
        """Doctor CSV export must only include scans belonging to assigned patients."""
        self.client.login(username='dr_reports', password='doctorpassword')
        response = self.client.get(reverse('reports:export_csv'))
        csv_content = response.content.decode('utf-8')

        self.assertIn(self.assigned_patient.patient_code, csv_content)
        self.assertIn('James Kirk', csv_content)
        self.assertNotIn(self.unassigned_patient.patient_code, csv_content)
        self.assertNotIn('Spock', csv_content)

    def test_csv_export_respects_role_filtering_for_admin(self):
        """Admin CSV export must include scans for all patients system-wide."""
        self.client.login(username='admin_reports', password='adminpassword')
        response = self.client.get(reverse('reports:export_csv'))
        csv_content = response.content.decode('utf-8')

        self.assertIn(self.assigned_patient.patient_code, csv_content)
        self.assertIn('James Kirk', csv_content)
        self.assertIn(self.unassigned_patient.patient_code, csv_content)
        self.assertIn('Spock', csv_content)

    def test_csv_export_status_filter(self):
        """CSV export should filter records when status query parameter is provided."""
        # Create a scan with NEEDS_REVIEW
        scan_review = MRIScan.objects.create(
            patient=self.assigned_patient,
            image=create_test_image('review.jpg')
        )
        Prediction.objects.create(
            scan=scan_review,
            predicted_label='glioma',
            confidence=0.65,
            status='NEEDS_REVIEW'
        )

        self.client.login(username='admin_reports', password='adminpassword')
        response = self.client.get(reverse('reports:export_csv') + '?status=NEEDS_REVIEW')
        csv_content = response.content.decode('utf-8')

        self.assertIn('NEEDS_REVIEW', csv_content)
        self.assertNotIn('AUTO_ACCEPTED', csv_content)

    def test_pdf_generation_returns_200_and_pdf_content_type(self):
        """PDF generation must return HTTP 200 with application/pdf Content-Type."""
        self.client.login(username='dr_reports', password='doctorpassword')
        response = self.client.get(reverse('reports:generate_pdf', kwargs={'scan_id': self.assigned_scan.pk}))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['Content-Type'], 'application/pdf')
        # PDF documents start with %PDF- header magic bytes
        self.assertTrue(response.content.startswith(b'%PDF-'))
        self.assertIn(f'NeuroLens_Report_Scan_{self.assigned_scan.id}.pdf', response['Content-Disposition'])

    def test_pdf_generation_denied_for_unassigned_doctor(self):
        """Doctor attempting to generate a report for an unassigned patient must be denied."""
        self.client.login(username='dr_reports', password='doctorpassword')
        response = self.client.get(reverse('reports:generate_pdf', kwargs={'scan_id': self.unassigned_scan.pk}))
        self.assertRedirects(response, reverse('scans:dashboard'))

        messages = list(get_messages(response.wsgi_request))
        self.assertTrue(any('Access denied' in str(m) for m in messages))

    def test_pdf_generation_allowed_for_admin_on_any_patient(self):
        """Admin must be allowed to generate PDF reports for any patient scan."""
        self.client.login(username='admin_reports', password='adminpassword')
        response = self.client.get(reverse('reports:generate_pdf', kwargs={'scan_id': self.unassigned_scan.pk}))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['Content-Type'], 'application/pdf')
        self.assertTrue(response.content.startswith(b'%PDF-'))
