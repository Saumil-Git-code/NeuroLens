"""
Management command to seed demo data for the NeuroLens medical AI platform.

Seeds:
- Admin superuser (admin / admin123)
- Two clinical doctor users:
    * dr_sharma (Neuroradiology) / doctor123
    * dr_verma (Neurosurgery) / doctor123
- 10 dummy patients with obviously fake '(TEST)' markers
- CareTeam assignments (patients 1-6 to Dr. Sharma, patients 5-10 to Dr. Verma)
- Optional MRI scans and AI predictions if demo_images/ exists with class folders

Safe to re-run multiple times (idempotent).
"""

from pathlib import Path
import random

from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from django.conf import settings
from django.core.files import File

# NeuroLens model imports
from accounts.models import Profile
from patients.models import Patient, CareTeam
from scans.models import MRIScan, Prediction


class Command(BaseCommand):
    help = 'Seeds demo users, dummy patients, care teams, and optional MRI scans for NeuroLens.'

    def handle(self, *args, **options):
        self.stdout.write("Starting NeuroLens demo data seeding...\n")

        # Track creation counts for the end-of-run summary
        stats = {
            'users_created': 0,
            'users_updated': 0,
            'patients_created': 0,
            'patients_skipped': 0,
            'careteam_assignments': 0,
            'scans_created': 0,
        }

        # ---------------------------------------------------------------------
        # 1. Admin Superuser
        # ---------------------------------------------------------------------
        self.stdout.write("1. Setting up Admin superuser...")
        admin_user, admin_created = User.objects.get_or_create(
            username='admin',
            defaults={
                'first_name': 'System',
                'last_name': 'Administrator',
                'is_staff': True,
                'is_superuser': True,
            }
        )
        if admin_created:
            admin_user.set_password('admin123')
            admin_user.save()
            stats['users_created'] += 1
            self.stdout.write(self.style.SUCCESS("   Created admin superuser 'admin'."))
        else:
            admin_user.first_name = 'System'
            admin_user.last_name = 'Administrator'
            admin_user.is_staff = True
            admin_user.is_superuser = True
            admin_user.set_password('admin123')
            admin_user.save()
            stats['users_updated'] += 1
            self.stdout.write("   Updated existing admin user 'admin'.")

        # Update Admin profile (post_save signal on User ensures profile exists)
        admin_profile, _ = Profile.objects.get_or_create(user=admin_user)
        admin_profile.role = 'ADMIN'
        admin_profile.save()

        # ---------------------------------------------------------------------
        # 2. Doctor Users
        # ---------------------------------------------------------------------
        self.stdout.write("\n2. Setting up Doctor users...")
        doctors_data = [
            {
                'username': 'dr_sharma',
                'password': 'doctor123',
                'first_name': 'Priya',
                'last_name': 'Sharma',
                'specialization': 'Neuroradiology',
                'phone': '+91-9876543210',
            },
            {
                'username': 'dr_verma',
                'password': 'doctor123',
                'first_name': 'Rahul',
                'last_name': 'Verma',
                'specialization': 'Neurosurgery',
                'phone': '+91-9876543211',
            },
        ]

        doctors = {}
        for doc_info in doctors_data:
            doc_user, created = User.objects.get_or_create(
                username=doc_info['username'],
                defaults={
                    'first_name': doc_info['first_name'],
                    'last_name': doc_info['last_name'],
                }
            )
            doc_user.first_name = doc_info['first_name']
            doc_user.last_name = doc_info['last_name']
            doc_user.set_password(doc_info['password'])
            doc_user.save()

            if created:
                stats['users_created'] += 1
                self.stdout.write(self.style.SUCCESS(
                    f"   Created doctor user '{doc_info['username']}' ({doc_info['first_name']} {doc_info['last_name']})."
                ))
            else:
                stats['users_updated'] += 1
                self.stdout.write(
                    f"   Updated existing doctor user '{doc_info['username']}'."
                )

            # Profile creation/update
            doc_profile, _ = Profile.objects.get_or_create(user=doc_user)
            doc_profile.role = 'DOCTOR'
            doc_profile.specialization = doc_info['specialization']
            doc_profile.phone = doc_info['phone']
            doc_profile.save()

            doctors[doc_info['username']] = doc_user

        dr_sharma = doctors['dr_sharma']
        dr_verma = doctors['dr_verma']

        # ---------------------------------------------------------------------
        # 3. 10 Dummy Patients with Obviously Fake Data
        # ---------------------------------------------------------------------
        self.stdout.write("\n3. Setting up dummy patients...")
        patients_data = [
            {'full_name': 'Aarav Kumar (TEST)', 'age': 45, 'gender': 'M', 'contact': '+91-0000000001', 'medical_history': 'Demo patient - chronic headaches, family history of neurological conditions.'},
            {'full_name': 'Diya Patel (TEST)', 'age': 32, 'gender': 'F', 'contact': '+91-0000000002', 'medical_history': 'Demo patient - visual disturbances, dizziness reported for 3 months.'},
            {'full_name': 'Vivaan Singh (TEST)', 'age': 58, 'gender': 'M', 'contact': '+91-0000000003', 'medical_history': 'Demo patient - seizures, memory loss, referred by general physician.'},
            {'full_name': 'Ananya Reddy (TEST)', 'age': 27, 'gender': 'F', 'contact': '+91-0000000004', 'medical_history': 'Demo patient - persistent migraines, no prior imaging.'},
            {'full_name': 'Arjun Gupta (TEST)', 'age': 63, 'gender': 'M', 'contact': '+91-0000000005', 'medical_history': 'Demo patient - weakness in left limbs, speech difficulties.'},
            {'full_name': 'Ishita Nair (TEST)', 'age': 41, 'gender': 'F', 'contact': '+91-0000000006', 'medical_history': 'Demo patient - hormonal imbalance, endocrine workup abnormal.'},
            {'full_name': 'Kabir Joshi (TEST)', 'age': 55, 'gender': 'M', 'contact': '+91-0000000007', 'medical_history': 'Demo patient - routine screening, asymptomatic.'},
            {'full_name': 'Meera Iyer (TEST)', 'age': 38, 'gender': 'F', 'contact': '+91-0000000008', 'medical_history': 'Demo patient - post-surgical follow-up, previous tumor resection.'},
            {'full_name': 'Rohan Das (TEST)', 'age': 50, 'gender': 'M', 'contact': '+91-0000000009', 'medical_history': 'Demo patient - cognitive decline, personality changes observed.'},
            {'full_name': 'Siya Malhotra (TEST)', 'age': 29, 'gender': 'F', 'contact': '+91-0000000010', 'medical_history': 'Demo patient - incidental finding on head CT after accident.'},
        ]

        patients = []
        # Check if patients with '(TEST)' already exist to avoid duplicate patient records
        if Patient.objects.filter(full_name__contains='(TEST)').exists():
            self.stdout.write(self.style.WARNING(
                "   Demo patients with '(TEST)' already exist. Skipping patient creation."
            ))
            # Retrieve existing test patients to maintain list for CareTeam assignments and scans
            for p_info in patients_data:
                existing = Patient.objects.filter(full_name=p_info['full_name']).first()
                if existing:
                    patients.append(existing)

            # Fallback if names differ slightly
            if len(patients) < 10:
                additional = list(
                    Patient.objects.filter(full_name__contains='(TEST)')
                    .exclude(id__in=[p.id for p in patients])
                    .order_by('id')
                )
                patients.extend(additional[:10 - len(patients)])

            stats['patients_skipped'] = len(patients)
        else:
            for p_info in patients_data:
                patient = Patient.objects.create(
                    full_name=p_info['full_name'],
                    age=p_info['age'],
                    gender=p_info['gender'],
                    contact=p_info['contact'],
                    medical_history=p_info['medical_history'],
                    created_by=admin_user,
                )
                patients.append(patient)
                stats['patients_created'] += 1
                self.stdout.write(f"   Created patient: {patient.patient_code} - {patient.full_name}")

        # ---------------------------------------------------------------------
        # 4. CareTeam Assignments
        # ---------------------------------------------------------------------
        # Patients 1-6 (indexes 0 to 5) assigned to Dr. Sharma
        # Patients 5-10 (indexes 4 to 9) assigned to Dr. Verma
        # Patients 5 and 6 (indexes 4 and 5) are shared between both doctors
        self.stdout.write("\n4. Assigning patients to CareTeams...")
        if len(patients) >= 10:
            # Patients 1-6 to Dr. Sharma
            for patient in patients[0:6]:
                _, created = CareTeam.objects.get_or_create(doctor=dr_sharma, patient=patient)
                if created:
                    stats['careteam_assignments'] += 1

            # Patients 5-10 to Dr. Verma
            for patient in patients[4:10]:
                _, created = CareTeam.objects.get_or_create(doctor=dr_verma, patient=patient)
                if created:
                    stats['careteam_assignments'] += 1

            self.stdout.write(self.style.SUCCESS(
                f"   CareTeam assigned: Patients 1-6 -> Dr. Sharma; Patients 5-10 -> Dr. Verma (New: {stats['careteam_assignments']})."
            ))
        else:
            self.stdout.write(self.style.WARNING(
                f"   Notice: Expected 10 patients, found {len(patients)}. Assigned available patients."
            ))
            for i, patient in enumerate(patients):
                if i < 6:
                    CareTeam.objects.get_or_create(doctor=dr_sharma, patient=patient)
                if i >= 4:
                    CareTeam.objects.get_or_create(doctor=dr_verma, patient=patient)

        # ---------------------------------------------------------------------
        # 5. Optional Demo Images & Predictions
        # ---------------------------------------------------------------------
        self.stdout.write("\n5. Checking for demo MRI images...")
        demo_dir = Path(settings.BASE_DIR) / 'demo_images'
        expected_classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

        images_by_class = {}
        if demo_dir.is_dir():
            for cls_name in expected_classes:
                cls_folder = demo_dir / cls_name
                if cls_folder.is_dir():
                    images = [
                        img_path for img_path in cls_folder.iterdir()
                        if img_path.is_file() and img_path.suffix.lower() in valid_extensions
                    ]
                    if images:
                        images_by_class[cls_name] = images

        if not images_by_class:
            self.stdout.write(self.style.WARNING(
                "   demo_images/ directory not found or empty (requires subfolders: glioma, meningioma, notumor, pituitary). Skipping scan seeding."
            ))
        else:
            self.stdout.write(f"   Found demo images for classes: {', '.join(images_by_class.keys())}")
            try:
                from ml.predictor import get_prediction
            except ImportError as e:
                self.stdout.write(self.style.ERROR(f"   Unable to import ml.predictor: {e}. Skipping scans."))
                get_prediction = None

            if get_prediction is not None:
                available_classes = list(images_by_class.keys())
                confidence_threshold = getattr(settings, 'NEURO_CONFIDENCE_THRESHOLD', 0.80)

                for patient in patients:
                    # Skip if patient already has scans to avoid duplicates on multiple runs
                    if patient.scans.exists():
                        continue

                    chosen_class = random.choice(available_classes)
                    chosen_image_file = random.choice(images_by_class[chosen_class])

                    # Create MRIScan and save image to media directory
                    scan = MRIScan(
                        patient=patient,
                        uploaded_by=admin_user,
                        notes=f"Demo scan seeded from class '{chosen_class}' ({chosen_image_file.name}).",
                    )
                    with open(chosen_image_file, 'rb') as f:
                        scan.image.save(chosen_image_file.name, File(f), save=True)

                    # Execute prediction using ML pipeline
                    try:
                        pred_result = get_prediction(scan.image.path)
                        conf = float(pred_result.get('confidence', 0.0))
                        is_mock = bool(pred_result.get('is_mock', False))
                        label = pred_result.get('label') or pred_result.get('predicted_label', chosen_class)

                        # Set status using NEURO_CONFIDENCE_THRESHOLD
                        status = 'AUTO_ACCEPTED' if conf >= confidence_threshold else 'NEEDS_REVIEW'

                        prediction = Prediction.objects.create(
                            scan=scan,
                            predicted_label=label,
                            confidence=conf,
                            probabilities=pred_result.get('probabilities', {}),
                            model_name=pred_result.get('model_name', 'vgg16'),
                            is_mock=is_mock,
                            status=status,
                        )

                        # Attach Grad-CAM heatmap if available
                        heatmap_path = pred_result.get('heatmap_path')
                        if heatmap_path:
                            prediction.heatmap_image = heatmap_path
                            prediction.save(update_fields=['heatmap_image'])

                        stats['scans_created'] += 1
                        self.stdout.write(
                            f"   Created Scan #{scan.id} for {patient.patient_code} -> "
                            f"{label} ({conf:.1%}) [{status}]"
                        )
                    except Exception as err:
                        self.stdout.write(self.style.ERROR(
                            f"   Error predicting scan for patient {patient.patient_code}: {err}"
                        ))

        # ---------------------------------------------------------------------
        # 6. Seeding Summary
        # ---------------------------------------------------------------------
        self.stdout.write("\n" + "=" * 60)
        self.stdout.write(self.style.SUCCESS("NeuroLens Demo Seeding Complete! Summary:"))
        self.stdout.write(f" - Users created:             {stats['users_created']}")
        self.stdout.write(f" - Users updated:             {stats['users_updated']}")
        self.stdout.write(f" - Patients created:          {stats['patients_created']}")
        self.stdout.write(f" - Patients skipped (exist):  {stats['patients_skipped']}")
        self.stdout.write(f" - CareTeam assignments:      {stats['careteam_assignments']}")
        self.stdout.write(f" - MRI Scans seeded:          {stats['scans_created']}")
        self.stdout.write("=" * 60 + "\n")
