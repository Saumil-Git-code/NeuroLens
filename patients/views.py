from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.core.paginator import Paginator
from django.db.models import Q
from .models import Patient, CareTeam
from .forms import PatientForm


def can_access_patient(user, patient):
    """Check if user can access this patient.
    
    Admins can access all patients. Doctors can only access assigned patients.
    """
    if hasattr(user, 'profile') and user.profile.is_admin:
        return True
    if user.is_superuser:
        return True
    return CareTeam.objects.filter(doctor=user, patient=patient).exists()


@login_required
def patient_list(request):
    """List patients accessible by the current user with search and pagination."""
    query = request.GET.get('q', '').strip()
    user = request.user
    is_admin = (hasattr(user, 'profile') and user.profile.is_admin) or user.is_superuser

    if is_admin:
        patients_qs = Patient.objects.all()
    else:
        patients_qs = Patient.objects.filter(care_team__doctor=user).distinct()

    if query:
        patients_qs = patients_qs.filter(
            Q(patient_code__icontains=query) |
            Q(full_name__icontains=query) |
            Q(contact__icontains=query)
        )

    paginator = Paginator(patients_qs, 12)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    context = {
        'page_obj': page_obj,
        'query': query,
        'total_count': patients_qs.count(),
        'is_admin': is_admin,
    }
    return render(request, 'patients/patient_list.html', context)


@login_required
def patient_create(request):
    """Create a new patient record and auto-assign care team for doctors."""
    if request.method == 'POST':
        form = PatientForm(request.POST)
        if form.is_valid():
            patient = form.save(commit=False)
            patient.created_by = request.user
            patient.save()

            # If user is a doctor, automatically add them to the patient's care team
            is_doctor = hasattr(request.user, 'profile') and request.user.profile.is_doctor
            if is_doctor or not (hasattr(request.user, 'profile') and request.user.profile.is_admin):
                CareTeam.objects.get_or_create(doctor=request.user, patient=patient)

            messages.success(request, f'Patient {patient.full_name} ({patient.patient_code}) created successfully.')
            return redirect('patients:detail', pk=patient.pk)
    else:
        form = PatientForm()

    return render(request, 'patients/patient_form.html', {
        'form': form,
        'title': 'Register New Patient',
        'is_edit': False,
    })


@login_required
def patient_detail(request, pk):
    """View patient details, associated MRI scans, and care team members."""
    patient = get_object_or_404(Patient, pk=pk)

    if not can_access_patient(request.user, patient):
        messages.error(request, 'Access denied. You are not assigned to this patient.')
        return redirect('patients:list')

    scans = patient.scans.select_related('prediction').all().order_by('-uploaded_at')
    care_team = patient.care_team.select_related('doctor', 'doctor__profile').all()

    context = {
        'patient': patient,
        'scans': scans,
        'care_team': care_team,
        'is_admin': (hasattr(request.user, 'profile') and request.user.profile.is_admin) or request.user.is_superuser,
    }
    return render(request, 'patients/patient_detail.html', context)


@login_required
def patient_edit(request, pk):
    """Edit patient information."""
    patient = get_object_or_404(Patient, pk=pk)

    if not can_access_patient(request.user, patient):
        messages.error(request, 'Access denied. You cannot edit this patient.')
        return redirect('patients:list')

    if request.method == 'POST':
        form = PatientForm(request.POST, instance=patient)
        if form.is_valid():
            form.save()
            messages.success(request, f'Patient record for {patient.full_name} updated successfully.')
            return redirect('patients:detail', pk=patient.pk)
    else:
        form = PatientForm(instance=patient)

    return render(request, 'patients/patient_form.html', {
        'form': form,
        'patient': patient,
        'title': f'Edit Patient: {patient.full_name}',
        'is_edit': True,
    })


@login_required
def patient_delete(request, pk):
    """Delete a patient record (Admin only)."""
    patient = get_object_or_404(Patient, pk=pk)
    is_admin = (hasattr(request.user, 'profile') and request.user.profile.is_admin) or request.user.is_superuser

    if not is_admin:
        messages.error(request, 'Permission denied: Only administrators can delete patient records.')
        return redirect('patients:detail', pk=patient.pk)

    if request.method == 'POST':
        name = patient.full_name
        code = patient.patient_code
        patient.delete()
        messages.success(request, f'Patient {name} ({code}) has been deleted.')
        return redirect('patients:list')

    return render(request, 'patients/patient_confirm_delete.html', {'patient': patient})
