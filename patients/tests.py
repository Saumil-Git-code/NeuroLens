"""
Tests for NeuroLens Patients Application.
=========================================
Covers Patient model auto-coding, age validation, CareTeam relationships,
role-based clinical data isolation (can_access_patient), view-level filtering,
and administrator-only delete permissions.
"""

from django.test import TestCase
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError
from django.db import IntegrityError
from django.urls import reverse
from django.contrib.messages import get_messages

from accounts.models import Profile
from patients.models import Patient, CareTeam
from patients.views import can_access_patient
from patients.forms import PatientForm


class PatientModelTests(TestCase):
    """Unit tests for the Patient model and auto-generated fields."""

    def test_patient_auto_generated_code_format(self):
        """First patient created without a code receives 'NL-0001'."""
        patient = Patient.objects.create(
            full_name='Alice Smith',
            age=45,
            gender='F',
            contact='555-0101'
        )
        self.assertEqual(patient.patient_code, 'NL-0001')

    def test_patient_auto_generated_code_sequential(self):
        """Subsequent patients receive sequential formatted codes."""
        p1 = Patient.objects.create(full_name='Alice Smith', age=45, gender='F')
        p2 = Patient.objects.create(full_name='Bob Jones', age=52, gender='M')
        p3 = Patient.objects.create(full_name='Charlie Brown', age=30, gender='M')

        self.assertEqual(p1.patient_code, 'NL-0001')
        self.assertEqual(p2.patient_code, 'NL-0002')
        self.assertEqual(p3.patient_code, 'NL-0003')

    def test_patient_code_collision_handling(self):
        """If a patient code already exists, the save method finds the next available code."""
        # Manually create NL-0002 first
        Patient.objects.create(patient_code='NL-0002', full_name='Existing Code', age=40, gender='F')
        # Create a patient without code; initial calculation might yield NL-0002 based on id, but should avoid collision
        p = Patient.objects.create(full_name='Next Patient', age=35, gender='M')
        self.assertNotEqual(p.patient_code, 'NL-0002')
        self.assertTrue(p.patient_code.startswith('NL-'))

    def test_patient_age_validation_boundaries(self):
        """Ages 0 and 120 should be accepted as valid boundaries."""
        infant = Patient(full_name='Infant Patient', age=0, gender='M')
        elderly = Patient(full_name='Elderly Patient', age=120, gender='F')

        # full_clean() should succeed without raising ValidationError
        infant.full_clean()
        elderly.full_clean()
        infant.save()
        elderly.save()
        self.assertEqual(infant.age, 0)
        self.assertEqual(elderly.age, 120)

    def test_patient_age_validation_negative_rejected(self):
        """Negative ages must fail validation."""
        patient = Patient(full_name='Invalid Infant', age=-1, gender='M')
        with self.assertRaises(ValidationError):
            patient.full_clean()

    def test_patient_age_validation_over_120_rejected(self):
        """Ages exceeding 120 must fail validation."""
        patient = Patient(full_name='Supercentenarian', age=121, gender='F')
        with self.assertRaises(ValidationError):
            patient.full_clean()

    def test_patient_form_rejects_invalid_age(self):
        """PatientForm validation must reject ages outside the 0-120 range."""
        form_too_old = PatientForm(data={
            'full_name': 'Invalid Age Patient',
            'age': 130,
            'gender': 'M',
        })
        self.assertFalse(form_too_old.is_valid())
        self.assertIn('age', form_too_old.errors)

    def test_patient_str_representation(self):
        """__str__ should return 'NL-XXXX - Full Name'."""
        patient = Patient.objects.create(
            full_name='Diana Prince',
            age=32,
            gender='F'
        )
        self.assertEqual(str(patient), f"{patient.patient_code} - Diana Prince")


class CareTeamModelTests(TestCase):
    """Unit tests for CareTeam assignments and constraints."""

    def setUp(self):
        self.doctor = User.objects.create_user(
            username='dr_watson',
            password='password123',
            first_name='John',
            last_name='Watson'
        )
        self.patient = Patient.objects.create(
            full_name='Sherlock Holmes',
            age=38,
            gender='M'
        )

    def test_care_team_relationship_creation(self):
        """CareTeam successfully links a doctor and a patient."""
        assignment = CareTeam.objects.create(doctor=self.doctor, patient=self.patient)
        self.assertEqual(self.doctor.care_assignments.count(), 1)
        self.assertEqual(self.patient.care_team.count(), 1)
        self.assertEqual(self.patient.care_team.first().doctor, self.doctor)
        self.assertEqual(str(assignment), "Dr. John Watson -> Sherlock Holmes")

    def test_care_team_unique_together_constraint(self):
        """A doctor cannot be assigned to the same patient more than once."""
        CareTeam.objects.create(doctor=self.doctor, patient=self.patient)
        with self.assertRaises(IntegrityError):
            CareTeam.objects.create(doctor=self.doctor, patient=self.patient)


class PatientAccessPermissionTests(TestCase):
    """Tests for the can_access_patient helper function."""

    def setUp(self):
        # Admin user
        self.admin_user = User.objects.create_superuser(
            username='admin_user',
            password='adminpassword'
        )
        # Doctor user
        self.doctor_user = User.objects.create_user(
            username='dr_primary',
            password='doctorpassword'
        )
        # Another doctor user
        self.other_doctor = User.objects.create_user(
            username='dr_secondary',
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
            age=55,
            gender='F'
        )
        # Assign CareTeam to primary doctor
        CareTeam.objects.create(doctor=self.doctor_user, patient=self.assigned_patient)

    def test_can_access_patient_admin_can_access_any_patient(self):
        """Admin users must have access to both assigned and unassigned patients."""
        self.assertTrue(can_access_patient(self.admin_user, self.assigned_patient))
        self.assertTrue(can_access_patient(self.admin_user, self.unassigned_patient))

    def test_can_access_patient_doctor_can_only_access_assigned_patients(self):
        """Doctors must only be granted access to patients in their CareTeam."""
        self.assertTrue(can_access_patient(self.doctor_user, self.assigned_patient))
        self.assertFalse(can_access_patient(self.doctor_user, self.unassigned_patient))
        # Other doctor has access to neither
        self.assertFalse(can_access_patient(self.other_doctor, self.assigned_patient))


class PatientViewsTests(TestCase):
    """Functional tests for patient management views and role filtering."""

    def setUp(self):
        # Admin user
        self.admin_user = User.objects.create_superuser(
            username='chief_admin',
            password='adminpassword'
        )
        # Doctor user
        self.doctor_user = User.objects.create_user(
            username='attending_physician',
            password='doctorpassword'
        )
        # Assigned patient
        self.assigned_patient = Patient.objects.create(
            full_name='Assigned Patient',
            age=45,
            gender='M',
            contact='111-222-3333'
        )
        # Unassigned patient
        self.unassigned_patient = Patient.objects.create(
            full_name='Unassigned Patient',
            age=60,
            gender='F',
            contact='444-555-6666'
        )
        # Assign CareTeam
        CareTeam.objects.create(doctor=self.doctor_user, patient=self.assigned_patient)

    def test_patient_list_view_filters_by_role_for_doctor(self):
        """Doctor viewing patient list must only see assigned patients."""
        self.client.login(username='attending_physician', password='doctorpassword')
        response = self.client.get(reverse('patients:list'))
        self.assertEqual(response.status_code, 200)

        patients_in_context = list(response.context['page_obj'])
        self.assertIn(self.assigned_patient, patients_in_context)
        self.assertNotIn(self.unassigned_patient, patients_in_context)

    def test_patient_list_view_shows_all_for_admin(self):
        """Admin viewing patient list must see all patients system-wide."""
        self.client.login(username='chief_admin', password='adminpassword')
        response = self.client.get(reverse('patients:list'))
        self.assertEqual(response.status_code, 200)

        patients_in_context = list(response.context['page_obj'])
        self.assertIn(self.assigned_patient, patients_in_context)
        self.assertIn(self.unassigned_patient, patients_in_context)

    def test_patient_create_creates_care_team_assignment_for_doctors(self):
        """Creating a patient as a doctor should automatically create a CareTeam link."""
        self.client.login(username='attending_physician', password='doctorpassword')
        response = self.client.post(reverse('patients:create'), {
            'full_name': 'Newly Created Patient',
            'age': 29,
            'gender': 'F',
            'contact': '999-888-7777',
            'medical_history': 'No prior neurological episodes.'
        })
        new_patient = Patient.objects.get(full_name='Newly Created Patient')
        self.assertRedirects(response, reverse('patients:detail', kwargs={'pk': new_patient.pk}))

        # Verify auto-assigned care team
        has_care_assignment = CareTeam.objects.filter(
            doctor=self.doctor_user,
            patient=new_patient
        ).exists()
        self.assertTrue(has_care_assignment)

    def test_patient_detail_denies_unassigned_doctor(self):
        """Doctor attempting to view unassigned patient must be redirected with an error message."""
        self.client.login(username='attending_physician', password='doctorpassword')
        response = self.client.get(reverse('patients:detail', kwargs={'pk': self.unassigned_patient.pk}))
        self.assertRedirects(response, reverse('patients:list'))

        messages = list(get_messages(response.wsgi_request))
        self.assertTrue(any('Access denied' in str(m) for m in messages))

    def test_patient_detail_allows_assigned_doctor(self):
        """Doctor viewing assigned patient detail should succeed."""
        self.client.login(username='attending_physician', password='doctorpassword')
        response = self.client.get(reverse('patients:detail', kwargs={'pk': self.assigned_patient.pk}))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'patients/patient_detail.html')
        self.assertEqual(response.context['patient'], self.assigned_patient)

    def test_patient_detail_allows_admin_for_any_patient(self):
        """Admin viewing unassigned patient detail should succeed."""
        self.client.login(username='chief_admin', password='adminpassword')
        response = self.client.get(reverse('patients:detail', kwargs={'pk': self.unassigned_patient.pk}))
        self.assertEqual(response.status_code, 200)

    def test_patient_delete_is_admin_only_doctor_denied(self):
        """Doctors cannot delete patient records."""
        self.client.login(username='attending_physician', password='doctorpassword')
        response = self.client.post(reverse('patients:delete', kwargs={'pk': self.assigned_patient.pk}))
        # Redirected back to detail with permission denied message
        self.assertRedirects(response, reverse('patients:detail', kwargs={'pk': self.assigned_patient.pk}))
        self.assertTrue(Patient.objects.filter(pk=self.assigned_patient.pk).exists())

        messages = list(get_messages(response.wsgi_request))
        self.assertTrue(any('Permission denied' in str(m) for m in messages))

    def test_patient_delete_is_admin_only_admin_allowed(self):
        """Admin can successfully delete patient records."""
        self.client.login(username='chief_admin', password='adminpassword')
        response = self.client.post(reverse('patients:delete', kwargs={'pk': self.assigned_patient.pk}))
        self.assertRedirects(response, reverse('patients:list'))
        self.assertFalse(Patient.objects.filter(pk=self.assigned_patient.pk).exists())
