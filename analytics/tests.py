"""
Tests for NeuroLens Analytics Application.
==========================================
Covers the analytics dashboard view, JSON data API endpoint,
key metrics calculation (class distribution, confidence histogram,
scan trends, clinician agreement), and role-based data partitioning.
"""

import io
import shutil
import tempfile
from PIL import Image as PILImage

from django.test import TestCase, override_settings
from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse

from accounts.models import Profile
from patients.models import Patient, CareTeam
from scans.models import MRIScan, Prediction, DoctorReview
from ml import CLASS_NAMES


def create_test_image(filename='test.jpg', size=(224, 224), format='JPEG'):
    """Create a simple test image in memory."""
    img = PILImage.new('RGB', size, color='gray')
    buffer = io.BytesIO()
    img.save(buffer, format=format)
    buffer.seek(0)
    return SimpleUploadedFile(filename, buffer.read(), content_type=f'image/{format.lower()}')


TEMP_MEDIA_DIR = tempfile.mkdtemp()


@override_settings(MEDIA_ROOT=TEMP_MEDIA_DIR)
class AnalyticsViewsTests(TestCase):
    """Integration and authorization tests for analytics dashboard and data feed API."""

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        shutil.rmtree(TEMP_MEDIA_DIR, ignore_errors=True)

    def setUp(self):
        # Admin user
        self.admin_user = User.objects.create_superuser(
            username='admin_analytics',
            password='adminpassword'
        )
        # Doctor user
        self.doctor_user = User.objects.create_user(
            username='dr_analytics',
            password='doctorpassword'
        )

        # Assigned patient
        self.assigned_patient = Patient.objects.create(
            full_name='Arthur Dent',
            age=42,
            gender='M'
        )
        # Unassigned patient
        self.unassigned_patient = Patient.objects.create(
            full_name='Ford Prefect',
            age=48,
            gender='O'
        )
        # Link CareTeam
        CareTeam.objects.create(doctor=self.doctor_user, patient=self.assigned_patient)

        # Scans
        self.assigned_scan = MRIScan.objects.create(
            patient=self.assigned_patient,
            image=create_test_image('arthur.jpg'),
            uploaded_by=self.doctor_user
        )
        self.unassigned_scan = MRIScan.objects.create(
            patient=self.unassigned_patient,
            image=create_test_image('ford.jpg'),
            uploaded_by=self.admin_user
        )

        # Predictions
        self.assigned_pred = Prediction.objects.create(
            scan=self.assigned_scan,
            predicted_label='glioma',
            confidence=0.85,
            probabilities={'glioma': 0.85, 'meningioma': 0.05, 'notumor': 0.05, 'pituitary': 0.05},
            status='REVIEWED'
        )
        self.unassigned_pred = Prediction.objects.create(
            scan=self.unassigned_scan,
            predicted_label='meningioma',
            confidence=0.65,
            probabilities={'glioma': 0.15, 'meningioma': 0.65, 'notumor': 0.10, 'pituitary': 0.10},
            status='NEEDS_REVIEW'
        )

        # Doctor Review on assigned scan
        self.assigned_review = DoctorReview.objects.create(
            prediction=self.assigned_pred,
            reviewer=self.doctor_user,
            final_label='glioma',
            agrees_with_ai=True,
            clinical_notes='AI diagnosis confirmed'
        )

    def test_analytics_dashboard_returns_200(self):
        """Analytics dashboard should return HTTP 200 with required context metrics."""
        self.client.login(username='dr_analytics', password='doctorpassword')
        response = self.client.get(reverse('analytics:dashboard'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'analytics/dashboard.html')

        self.assertIn('total_patients', response.context)
        self.assertIn('total_scans', response.context)
        self.assertIn('pending_count', response.context)
        self.assertIn('agreement_rate', response.context)

    def test_analytics_dashboard_accessible_to_admin(self):
        """Admin user can also access the analytics dashboard."""
        self.client.login(username='admin_analytics', password='adminpassword')
        response = self.client.get(reverse('analytics:dashboard'))
        self.assertEqual(response.status_code, 200)

    def test_analytics_data_api_returns_valid_json_with_expected_keys(self):
        """API endpoint should return HTTP 200 and JSON with all required visualization data feeds."""
        self.client.login(username='dr_analytics', password='doctorpassword')
        response = self.client.get(reverse('analytics:data_api'))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['Content-Type'], 'application/json')

        data = response.json()
        expected_keys = [
            'class_distribution',
            'scans_per_day',
            'confidence_histogram',
            'agreement',
            'review_table',
            'stats',
        ]
        for key in expected_keys:
            self.assertIn(key, data)

        # Check stats sub-keys
        stats = data['stats']
        for sub_key in ['total_patients', 'total_scans', 'pending_reviews', 'agreement_rate']:
            self.assertIn(sub_key, stats)

        # Check all 4 tumor classes are present in class_distribution
        for cls_name in CLASS_NAMES:
            self.assertIn(cls_name, data['class_distribution'])

    def test_analytics_data_respects_role_filtering_for_doctor(self):
        """Doctor should only receive analytics aggregated over their assigned patients."""
        self.client.login(username='dr_analytics', password='doctorpassword')
        response = self.client.get(reverse('analytics:data_api'))
        data = response.json()

        # Doctor only has 1 assigned patient and 1 assigned scan
        self.assertEqual(data['stats']['total_patients'], 1)
        self.assertEqual(data['stats']['total_scans'], 1)
        # Assigned prediction is REVIEWED, so pending_reviews for doctor is 0
        self.assertEqual(data['stats']['pending_reviews'], 0)

    def test_analytics_data_respects_role_filtering_for_admin(self):
        """Admin should receive analytics aggregated over all patients across the entire clinic."""
        self.client.login(username='admin_analytics', password='adminpassword')
        response = self.client.get(reverse('analytics:data_api'))
        data = response.json()

        # Admin sees both patients and both scans
        self.assertEqual(data['stats']['total_patients'], 2)
        self.assertEqual(data['stats']['total_scans'], 2)
        # Unassigned prediction has status NEEDS_REVIEW
        self.assertEqual(data['stats']['pending_reviews'], 1)

    def test_analytics_scans_per_day_length(self):
        """scans_per_day must contain 30 daily buckets for trend visualization."""
        self.client.login(username='dr_analytics', password='doctorpassword')
        response = self.client.get(reverse('analytics:data_api'))
        data = response.json()
        self.assertEqual(len(data['scans_per_day']), 30)

    def test_analytics_confidence_histogram_bins(self):
        """confidence_histogram should contain 5 percentage range bins."""
        self.client.login(username='dr_analytics', password='doctorpassword')
        response = self.client.get(reverse('analytics:data_api'))
        data = response.json()
        self.assertEqual(len(data['confidence_histogram']['labels']), 5)
        self.assertEqual(len(data['confidence_histogram']['counts']), 5)

    def test_unauthenticated_access_redirects_to_login(self):
        """Unauthenticated requests to analytics endpoints must redirect to login."""
        endpoints = [
            reverse('analytics:dashboard'),
            reverse('analytics:data_api'),
        ]
        for url in endpoints:
            with self.subTest(url=url):
                response = self.client.get(url)
                self.assertEqual(response.status_code, 302)
                self.assertIn(reverse('accounts:login'), response.url)
