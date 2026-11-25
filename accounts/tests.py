"""
Tests for NeuroLens Accounts Application.
=========================================
Covers User Profile creation, role-based properties, authentication flows
(login, logout), and redirection of unauthenticated access.
"""

from django.test import TestCase
from django.contrib.auth.models import User
from django.urls import reverse
from django.contrib.messages import get_messages

from accounts.models import Profile


class ProfileModelTests(TestCase):
    """Unit tests for the Profile model and its User post_save signal."""

    def test_profile_auto_created_on_user_creation(self):
        """Standard user creation should automatically create an associated Doctor profile."""
        user = User.objects.create_user(
            username='doctor_test',
            password='testpassword123',
            first_name='John',
            last_name='Watson'
        )
        self.assertTrue(hasattr(user, 'profile'))
        self.assertEqual(user.profile.user, user)
        self.assertEqual(user.profile.role, 'DOCTOR')

    def test_profile_auto_created_on_superuser_creation(self):
        """Superuser creation should automatically assign the ADMIN role to the Profile."""
        admin_user = User.objects.create_superuser(
            username='admin_test',
            password='adminpassword123',
            email='admin@neurolens.org'
        )
        self.assertTrue(hasattr(admin_user, 'profile'))
        self.assertEqual(admin_user.profile.role, 'ADMIN')

    def test_is_admin_property(self):
        """is_admin property must return True for ADMIN role and False for DOCTOR role."""
        admin_user = User.objects.create_superuser(
            username='admin_prop',
            password='password123'
        )
        doctor_user = User.objects.create_user(
            username='doctor_prop',
            password='password123'
        )
        self.assertTrue(admin_user.profile.is_admin)
        self.assertFalse(doctor_user.profile.is_admin)

    def test_is_doctor_property(self):
        """is_doctor property must return True for DOCTOR role and False for ADMIN role."""
        admin_user = User.objects.create_superuser(
            username='admin_prop2',
            password='password123'
        )
        doctor_user = User.objects.create_user(
            username='doctor_prop2',
            password='password123'
        )
        self.assertTrue(doctor_user.profile.is_doctor)
        self.assertFalse(admin_user.profile.is_doctor)

    def test_profile_str_representation(self):
        """__str__ method should return formatted name and role."""
        user = User.objects.create_user(
            username='jwatson',
            password='password123',
            first_name='John',
            last_name='Watson'
        )
        self.assertEqual(str(user.profile), 'John Watson (DOCTOR)')

        user_no_name = User.objects.create_user(
            username='jdoe_noname',
            password='password123'
        )
        self.assertEqual(str(user_no_name.profile), 'jdoe_noname (DOCTOR)')


class AuthenticationViewsTests(TestCase):
    """Functional tests for login, logout, and access control redirects."""

    def setUp(self):
        """Set up test users for authentication."""
        self.doctor_user = User.objects.create_user(
            username='dr_house',
            password='correct_password',
            first_name='Gregory',
            last_name='House'
        )
        self.admin_user = User.objects.create_superuser(
            username='admin_cuddy',
            password='admin_password',
            first_name='Lisa',
            last_name='Cuddy'
        )

    def test_login_view_renders_form(self):
        """GET request to login page should render the login template."""
        response = self.client.get(reverse('accounts:login'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'accounts/login.html')

    def test_login_view_redirects_if_already_authenticated(self):
        """Authenticated users visiting login view should be redirected to dashboard."""
        self.client.login(username='dr_house', password='correct_password')
        response = self.client.get(reverse('accounts:login'))
        self.assertRedirects(response, reverse('scans:dashboard'))

    def test_login_view_with_valid_credentials(self):
        """Valid credentials should authenticate user and redirect to dashboard."""
        response = self.client.post(reverse('accounts:login'), {
            'username': 'dr_house',
            'password': 'correct_password'
        })
        self.assertRedirects(response, reverse('scans:dashboard'))
        self.assertTrue(response.wsgi_request.user.is_authenticated)

    def test_login_view_with_next_parameter(self):
        """Valid login with a 'next' query parameter should redirect to that destination."""
        next_url = reverse('patients:list')
        response = self.client.post(f"{reverse('accounts:login')}?next={next_url}", {
            'username': 'dr_house',
            'password': 'correct_password'
        })
        self.assertRedirects(response, next_url)

    def test_login_view_with_invalid_credentials(self):
        """Invalid credentials should fail authentication, re-render form, and show error."""
        response = self.client.post(reverse('accounts:login'), {
            'username': 'dr_house',
            'password': 'wrong_password'
        })
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'accounts/login.html')
        self.assertFalse(response.wsgi_request.user.is_authenticated)

        messages = list(get_messages(response.wsgi_request))
        self.assertTrue(any('Invalid username or password' in str(m) for m in messages))

    def test_login_view_with_nonexistent_user(self):
        """Non-existent username should fail authentication and return error message."""
        response = self.client.post(reverse('accounts:login'), {
            'username': 'nonexistent_user',
            'password': 'any_password'
        })
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.wsgi_request.user.is_authenticated)

    def test_logout_redirects_to_login_page(self):
        """Logout view should end user session and redirect to accounts login."""
        self.client.login(username='dr_house', password='correct_password')
        response = self.client.get(reverse('accounts:logout'))
        self.assertRedirects(response, reverse('accounts:login'))
        self.assertFalse(response.wsgi_request.user.is_authenticated)

    def test_unauthenticated_access_redirects_to_login(self):
        """Protected views must redirect unauthenticated requests to the login page."""
        protected_urls = [
            reverse('scans:dashboard'),
            reverse('patients:list'),
            reverse('scans:upload'),
            reverse('reports:export_csv'),
            reverse('analytics:dashboard'),
        ]
        for url in protected_urls:
            with self.subTest(url=url):
                response = self.client.get(url)
                self.assertEqual(response.status_code, 302)
                self.assertIn(reverse('accounts:login'), response.url)
