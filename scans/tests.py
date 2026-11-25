"""
Tests for NeuroLens Scans Application.
======================================
Covers MRIScan model, Prediction model, DoctorReview model, file upload validations,
AI prediction pipeline execution, clinician review queue, threshold-based triage,
search API role filtering, and mock predictor selection.
"""

import io
import os
import shutil
import tempfile
from unittest.mock import patch
from PIL import Image as PILImage

from django.test import TestCase, override_settings
from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.db import IntegrityError
from django.urls import reverse
from django.contrib.messages import get_messages

from accounts.models import Profile
from patients.models import Patient, CareTeam
from scans.models import MRIScan, Prediction, DoctorReview
from scans.forms import ScanUploadForm, ReviewForm


def create_test_image(filename='test.jpg', size=(224, 224), format='JPEG'):
    """Create a simple test image in memory."""
    img = PILImage.new('RGB', size, color='gray')
    buffer = io.BytesIO()
    img.save(buffer, format=format)
    buffer.seek(0)
    return SimpleUploadedFile(filename, buffer.read(), content_type=f'image/{format.lower()}')


# Use a temporary directory for MEDIA_ROOT during testing to prevent polluting real storage
TEMP_MEDIA_DIR = tempfile.mkdtemp()


@override_settings(MEDIA_ROOT=TEMP_MEDIA_DIR)
class ScanModelTests(TestCase):
    """Unit tests for MRIScan, Prediction, and DoctorReview models."""

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        shutil.rmtree(TEMP_MEDIA_DIR, ignore_errors=True)

    def setUp(self):
        self.patient = Patient.objects.create(full_name='Test Patient', age=45, gender='M')
        self.doctor = User.objects.create_user(
            username='dr_tester', password='password123',
            first_name='Test', last_name='Doctor'
        )
        self.scan_file = create_test_image('scan1.jpg')

    def test_mriscan_creation_and_patient_foreign_key(self):
        """MRIScan should correctly link to a Patient and generate string representation."""
        scan = MRIScan.objects.create(
            patient=self.patient,
            image=self.scan_file,
            uploaded_by=self.doctor,
            notes='Baseline axial scan'
        )
        self.assertEqual(scan.patient, self.patient)
        self.assertEqual(self.patient.scans.first(), scan)
        self.assertEqual(scan.uploaded_by, self.doctor)
        self.assertIn(self.patient.patient_code, str(scan))

    def test_prediction_onetoone_with_mriscan(self):
        """Prediction must have a OneToOne relationship with MRIScan."""
        scan = MRIScan.objects.create(patient=self.patient, image=self.scan_file)
        prediction = Prediction.objects.create(
            scan=scan,
            predicted_label='glioma',
            confidence=0.88,
            probabilities={'glioma': 0.88, 'meningioma': 0.04, 'notumor': 0.04, 'pituitary': 0.04},
            status='AUTO_ACCEPTED'
        )
        self.assertEqual(scan.prediction, prediction)
        self.assertEqual(prediction.scan, scan)
        self.assertEqual(str(prediction), "Prediction: glioma (88.0%)")

        # Creating a second prediction for the same scan should violate OneToOne constraint
        with self.assertRaises(IntegrityError):
            Prediction.objects.create(
                scan=scan,
                predicted_label='meningioma',
                confidence=0.75
            )

    def test_doctor_review_onetoone_with_prediction(self):
        """DoctorReview must have a OneToOne relationship with Prediction."""
        scan = MRIScan.objects.create(patient=self.patient, image=self.scan_file)
        prediction = Prediction.objects.create(
            scan=scan,
            predicted_label='glioma',
            confidence=0.88,
            status='AUTO_ACCEPTED'
        )
        review = DoctorReview.objects.create(
            prediction=prediction,
            reviewer=self.doctor,
            final_label='glioma',
            agrees_with_ai=True,
            clinical_notes='Verified contrast enhancement'
        )
        self.assertEqual(prediction.review, review)
        self.assertEqual(review.prediction, prediction)
        # __str__ should include reviewer name (falls back to username if no full_name)
        review_str = str(review)
        self.assertIn('dr_tester', review_str.lower().replace(' ', '_') if not self.doctor.get_full_name() else review_str)
        # With first/last name set, should appear in the string
        self.assertIn('Test Doctor', review_str)

        # Duplicate review for same prediction must violate OneToOne constraint
        with self.assertRaises(IntegrityError):
            DoctorReview.objects.create(
                prediction=prediction,
                reviewer=self.doctor,
                final_label='meningioma',
                agrees_with_ai=False
            )


@override_settings(MEDIA_ROOT=TEMP_MEDIA_DIR)
class ScanUploadFormTests(TestCase):
    """Validation tests for the ScanUploadForm."""

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        shutil.rmtree(TEMP_MEDIA_DIR, ignore_errors=True)

    def setUp(self):
        self.patient = Patient.objects.create(full_name='Test Patient', age=50, gender='F')
        self.doctor = User.objects.create_user(username='dr_form_test', password='password123')
        CareTeam.objects.create(doctor=self.doctor, patient=self.patient)

    def test_form_validates_and_accepts_valid_jpg(self):
        """Valid JPEG image passes form validation."""
        valid_img = create_test_image('brain_mri.jpg')
        form = ScanUploadForm(
            data={'patient': self.patient.id, 'notes': 'Routine follow up'},
            files={'image': valid_img},
            user=self.doctor
        )
        self.assertTrue(form.is_valid())

    def test_form_rejects_txt_file_extension(self):
        """Files with non-image extensions (.txt) must be rejected."""
        txt_file = SimpleUploadedFile('notes.txt', b'This is plain text, not an MRI image.', content_type='text/plain')
        form = ScanUploadForm(
            data={'patient': self.patient.id},
            files={'image': txt_file},
            user=self.doctor
        )
        self.assertFalse(form.is_valid())
        self.assertIn('image', form.errors)
        # Check that at least one error message mentions the format issue
        image_errors = ' '.join(form.errors['image']).lower()
        self.assertTrue(
            'unsupported' in image_errors or 'format' in image_errors
            or 'valid image' in image_errors or 'upload a valid image' in image_errors,
            f"Expected format-related error but got: {form.errors['image']}"
        )

    def test_form_rejects_oversized_file(self):
        """Files larger than 16MB must be rejected."""
        # Create a valid-header JPEG that exceeds 16 MB
        img = PILImage.new('RGB', (224, 224), color='gray')
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG')
        # Pad to exceed 16 MB
        buffer.write(b'\x00' * (17 * 1024 * 1024 - buffer.tell()))
        buffer.seek(0)
        large_file = SimpleUploadedFile('huge_mri.jpg', buffer.read(), content_type='image/jpeg')
        form = ScanUploadForm(
            data={'patient': self.patient.id},
            files={'image': large_file},
            user=self.doctor
        )
        self.assertFalse(form.is_valid())
        self.assertIn('image', form.errors)
        # Check the error mentions size
        image_errors = ' '.join(form.errors['image']).lower()
        self.assertTrue(
            'too large' in image_errors or 'size' in image_errors or 'maximum' in image_errors,
            f"Expected size-related error but got: {form.errors['image']}"
        )


@override_settings(MEDIA_ROOT=TEMP_MEDIA_DIR)
class ScanViewsTests(TestCase):
    """Functional tests for scan upload, result display, triage thresholds, review queue, and search API."""

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        shutil.rmtree(TEMP_MEDIA_DIR, ignore_errors=True)

    def setUp(self):
        # Admin user
        self.admin_user = User.objects.create_superuser(
            username='admin_scan_test',
            password='adminpassword'
        )
        # Doctor user
        self.doctor_user = User.objects.create_user(
            username='dr_scan_test',
            password='doctorpassword'
        )
        # Patients
        self.assigned_patient = Patient.objects.create(
            full_name='Assigned Patient',
            age=40,
            gender='M'
        )
        self.unassigned_patient = Patient.objects.create(
            full_name='Unassigned Patient',
            age=50,
            gender='F'
        )
        # Assign CareTeam
        CareTeam.objects.create(doctor=self.doctor_user, patient=self.assigned_patient)

        # Baseline scans
        self.assigned_scan = MRIScan.objects.create(
            patient=self.assigned_patient,
            image=create_test_image('assigned.jpg'),
            uploaded_by=self.doctor_user
        )
        self.unassigned_scan = MRIScan.objects.create(
            patient=self.unassigned_patient,
            image=create_test_image('unassigned.jpg'),
            uploaded_by=self.admin_user
        )

    def test_dashboard_view_accessible(self):
        """Dashboard view returns 200 for authenticated doctor."""
        self.client.login(username='dr_scan_test', password='doctorpassword')
        response = self.client.get(reverse('scans:dashboard'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'scans/dashboard.html')

    @patch('ml.predictor.get_prediction')
    def test_upload_creates_prediction(self, mock_predict):
        """Uploading an MRI scan invokes AI inference and creates an associated Prediction."""
        mock_predict.return_value = {
            'label': 'glioma',
            'confidence': 0.92,
            'probabilities': {'glioma': 0.92, 'meningioma': 0.04, 'notumor': 0.02, 'pituitary': 0.02},
            'heatmap_path': None,
            'model_name': 'mock',
            'is_mock': True,
        }

        self.client.login(username='dr_scan_test', password='doctorpassword')
        image = create_test_image('new_upload.jpg')
        response = self.client.post(reverse('scans:upload'), {
            'patient': self.assigned_patient.id,
            'image': image,
            'notes': 'Test axial scan'
        })

        new_scan = MRIScan.objects.filter(patient=self.assigned_patient).latest('uploaded_at')
        self.assertRedirects(response, reverse('scans:result', kwargs={'pk': new_scan.pk}))
        self.assertTrue(hasattr(new_scan, 'prediction'))
        self.assertEqual(new_scan.prediction.predicted_label, 'glioma')
        self.assertEqual(new_scan.prediction.confidence, 0.92)

    @patch('ml.predictor.get_prediction')
    def test_threshold_logic_confidence_gte_80_auto_accepted(self, mock_predict):
        """Real prediction with confidence >= 0.80 should be marked AUTO_ACCEPTED."""
        mock_predict.return_value = {
            'label': 'meningioma',
            'confidence': 0.88,
            'probabilities': {'glioma': 0.05, 'meningioma': 0.88, 'notumor': 0.03, 'pituitary': 0.04},
            'heatmap_path': None,
            'model_name': 'vgg16',
            'is_mock': False,  # Real model
        }

        self.client.login(username='dr_scan_test', password='doctorpassword')
        image = create_test_image('high_conf.jpg')
        self.client.post(reverse('scans:upload'), {
            'patient': self.assigned_patient.id,
            'image': image,
        })

        scan = MRIScan.objects.filter(patient=self.assigned_patient).latest('uploaded_at')
        self.assertEqual(scan.prediction.status, 'AUTO_ACCEPTED')

    @patch('ml.predictor.get_prediction')
    def test_threshold_logic_confidence_lt_80_needs_review(self, mock_predict):
        """Real prediction with confidence < 0.80 should be marked NEEDS_REVIEW."""
        mock_predict.return_value = {
            'label': 'meningioma',
            'confidence': 0.72,
            'probabilities': {'glioma': 0.15, 'meningioma': 0.72, 'notumor': 0.08, 'pituitary': 0.05},
            'heatmap_path': None,
            'model_name': 'vgg16',
            'is_mock': False,
        }

        self.client.login(username='dr_scan_test', password='doctorpassword')
        image = create_test_image('low_conf.jpg')
        self.client.post(reverse('scans:upload'), {
            'patient': self.assigned_patient.id,
            'image': image,
        })

        scan = MRIScan.objects.filter(patient=self.assigned_patient).latest('uploaded_at')
        self.assertEqual(scan.prediction.status, 'NEEDS_REVIEW')

    @patch('ml.predictor.get_prediction')
    def test_threshold_logic_mock_predictions_always_need_review(self, mock_predict):
        """Mock predictions should always be marked NEEDS_REVIEW regardless of confidence."""
        mock_predict.return_value = {
            'label': 'pituitary',
            'confidence': 0.95,
            'probabilities': {'glioma': 0.02, 'meningioma': 0.01, 'notumor': 0.02, 'pituitary': 0.95},
            'heatmap_path': None,
            'model_name': 'mock',
            'is_mock': True,  # Mock model
        }

        self.client.login(username='dr_scan_test', password='doctorpassword')
        image = create_test_image('mock_scan.jpg')
        self.client.post(reverse('scans:upload'), {
            'patient': self.assigned_patient.id,
            'image': image,
        })

        scan = MRIScan.objects.filter(patient=self.assigned_patient).latest('uploaded_at')
        self.assertEqual(scan.prediction.status, 'NEEDS_REVIEW')

    def test_scan_result_view_denies_unassigned_doctor(self):
        """Doctor attempting to view result of unassigned patient's scan must be denied."""
        self.client.login(username='dr_scan_test', password='doctorpassword')
        response = self.client.get(reverse('scans:result', kwargs={'pk': self.unassigned_scan.pk}))
        self.assertRedirects(response, reverse('scans:dashboard'))

        messages = list(get_messages(response.wsgi_request))
        self.assertTrue(any('Access denied' in str(m) for m in messages))

    def test_scan_result_view_allows_assigned_doctor(self):
        """Doctor viewing result of assigned patient's scan should succeed."""
        Prediction.objects.create(
            scan=self.assigned_scan,
            predicted_label='glioma',
            confidence=0.85,
            status='AUTO_ACCEPTED'
        )
        self.client.login(username='dr_scan_test', password='doctorpassword')
        response = self.client.get(reverse('scans:result', kwargs={'pk': self.assigned_scan.pk}))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'scans/scan_result.html')

    def test_review_queue_shows_only_needs_review_items(self):
        """Review queue must only display predictions with status='NEEDS_REVIEW'."""
        # 1. Assigned scan needing review
        p_needs_review = Prediction.objects.create(
            scan=self.assigned_scan,
            predicted_label='glioma',
            confidence=0.65,
            status='NEEDS_REVIEW'
        )
        # 2. Another assigned scan auto-accepted
        scan2 = MRIScan.objects.create(
            patient=self.assigned_patient,
            image=create_test_image('scan2.jpg')
        )
        p_auto_accepted = Prediction.objects.create(
            scan=scan2,
            predicted_label='notumor',
            confidence=0.95,
            status='AUTO_ACCEPTED'
        )
        # 3. Unassigned scan needing review
        p_unassigned_needs_review = Prediction.objects.create(
            scan=self.unassigned_scan,
            predicted_label='meningioma',
            confidence=0.60,
            status='NEEDS_REVIEW'
        )

        self.client.login(username='dr_scan_test', password='doctorpassword')
        response = self.client.get(reverse('scans:review_queue'))
        self.assertEqual(response.status_code, 200)

        predictions_in_queue = list(response.context['predictions'])
        self.assertIn(p_needs_review, predictions_in_queue)
        self.assertNotIn(p_auto_accepted, predictions_in_queue)
        # Doctor should NOT see unassigned patient's scan in their review queue
        self.assertNotIn(p_unassigned_needs_review, predictions_in_queue)

    def test_review_submission_updates_status_to_reviewed(self):
        """Submitting a clinician review updates prediction status to REVIEWED and creates DoctorReview."""
        prediction = Prediction.objects.create(
            scan=self.assigned_scan,
            predicted_label='glioma',
            confidence=0.70,
            status='NEEDS_REVIEW'
        )

        self.client.login(username='dr_scan_test', password='doctorpassword')
        response = self.client.post(reverse('scans:review', kwargs={'pk': prediction.pk}), {
            'final_label': 'meningioma',
            'agrees_with_ai': False,
            'clinical_notes': 'Clinical review reveals meningioma characteristics instead.'
        })

        self.assertRedirects(response, reverse('scans:result', kwargs={'pk': self.assigned_scan.pk}))
        prediction.refresh_from_db()
        self.assertEqual(prediction.status, 'REVIEWED')

        review = DoctorReview.objects.get(prediction=prediction)
        self.assertEqual(review.reviewer, self.doctor_user)
        self.assertEqual(review.final_label, 'meningioma')
        self.assertFalse(review.agrees_with_ai)

    def test_search_api_returns_json_and_respects_role(self):
        """Search API returns expected JSON format and filters by doctor assignment."""
        self.client.login(username='dr_scan_test', password='doctorpassword')
        response = self.client.get(reverse('scans:search_api') + '?q=Patient')
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertIn('patients', data)
        self.assertIn('scans', data)
        # Doctor only sees assigned patient
        patient_codes = [p['code'] for p in data['patients']]
        self.assertIn(self.assigned_patient.patient_code, patient_codes)
        self.assertNotIn(self.unassigned_patient.patient_code, patient_codes)

        # Admin sees all patients
        self.client.login(username='admin_scan_test', password='adminpassword')
        response_admin = self.client.get(reverse('scans:search_api') + '?q=Patient')
        data_admin = response_admin.json()
        admin_patient_codes = [p['code'] for p in data_admin['patients']]
        self.assertIn(self.assigned_patient.patient_code, admin_patient_codes)
        self.assertIn(self.unassigned_patient.patient_code, admin_patient_codes)

    def test_search_api_empty_query(self):
        """Search API returns empty lists when query is blank."""
        self.client.login(username='dr_scan_test', password='doctorpassword')
        response = self.client.get(reverse('scans:search_api') + '?q=')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {'patients': [], 'scans': []})


class PredictorSelectionTests(TestCase):
    """Tests for predictor fallback selection logic (mock vs real model weights)."""

    def test_mock_predictor_when_no_weights(self):
        """When no weights file exists, load_model returns None."""
        from ml.predictor import load_model, predict_mock
        result = load_model()
        self.assertIsNone(result)  # No weights file

    def test_get_prediction_uses_mock_fallback(self):
        """get_prediction should produce a valid dictionary with is_mock=True when no weights exist."""
        from ml.predictor import get_prediction
        temp_img_path = os.path.join(TEMP_MEDIA_DIR, 'test_mock_inference.jpg')
        os.makedirs(TEMP_MEDIA_DIR, exist_ok=True)
        PILImage.new('RGB', (224, 224), color='gray').save(temp_img_path)

        result = get_prediction(temp_img_path)
        self.assertIsInstance(result, dict)
        self.assertIn('label', result)
        self.assertIn('confidence', result)
        self.assertIn('probabilities', result)
        self.assertTrue(result['is_mock'])
        self.assertEqual(result['model_name'], 'mock')
