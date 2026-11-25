import os
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.conf import settings
from django.http import JsonResponse
from django.core.files import File
from django.db.models import Avg, Count, Q

from .models import MRIScan, Prediction, DoctorReview
from .forms import ScanUploadForm, ReviewForm
from patients.models import Patient, CareTeam
from patients.views import can_access_patient


def get_user_accessible_scans(user):
    """Return queryset of MRIScans accessible to this user."""
    is_admin = (hasattr(user, 'profile') and user.profile.is_admin) or user.is_superuser
    if is_admin:
        return MRIScan.objects.all()
    return MRIScan.objects.filter(patient__care_team__doctor=user).distinct()


@login_required
def dashboard(request):
    """Main landing dashboard showing KPI cards, recent scans, and quick actions."""
    user = request.user
    is_admin = (hasattr(user, 'profile') and user.profile.is_admin) or user.is_superuser

    if is_admin:
        patient_count = Patient.objects.count()
        scans_qs = MRIScan.objects.all()
        pending_count = Prediction.objects.filter(status='NEEDS_REVIEW').count()
        reviews = DoctorReview.objects.all()
    else:
        patient_count = Patient.objects.filter(care_team__doctor=user).distinct().count()
        scans_qs = MRIScan.objects.filter(patient__care_team__doctor=user).distinct()
        pending_count = Prediction.objects.filter(
            status='NEEDS_REVIEW',
            scan__patient__care_team__doctor=user
        ).distinct().count()
        reviews = DoctorReview.objects.filter(prediction__scan__patient__care_team__doctor=user).distinct()

    total_scans = scans_qs.count()
    total_reviews = reviews.count()
    agreed_reviews = reviews.filter(agrees_with_ai=True).count()
    agreement_rate = (agreed_reviews / total_reviews * 100) if total_reviews > 0 else 0.0

    recent_scans = scans_qs.select_related('patient', 'prediction').order_by('-uploaded_at')[:10]

    context = {
        'patient_count': patient_count,
        'total_scans': total_scans,
        'pending_count': pending_count,
        'agreement_rate': round(agreement_rate, 1),
        'recent_scans': recent_scans,
        'is_admin': is_admin,
    }
    return render(request, 'scans/dashboard.html', context)


@login_required
def scan_upload(request):
    """Upload a brain MRI scan and immediately invoke the AI prediction pipeline."""
    initial_patient = request.GET.get('patient')
    initial_data = {}
    if initial_patient:
        initial_data['patient'] = initial_patient

    if request.method == 'POST':
        form = ScanUploadForm(request.POST, request.FILES, user=request.user)
        if form.is_valid():
            scan = form.save(commit=False)
            scan.uploaded_by = request.user
            scan.save()

            # Execute AI inference
            threshold = getattr(settings, 'NEURO_CONFIDENCE_THRESHOLD', 0.80)
            try:
                from ml.predictor import get_prediction
                pred_result = get_prediction(scan.image.path)
            except Exception:
                # Safe fallback if ML predictor is loading or weights pending
                pred_result = {
                    'predicted_label': 'notumor',
                    'confidence': 0.85,
                    'probabilities': {
                        'glioma': 0.05,
                        'meningioma': 0.05,
                        'notumor': 0.85,
                        'pituitary': 0.05
                    },
                    'heatmap_path': None,
                    'model_name': 'vgg16-demo',
                    'is_mock': True
                }

            conf = float(pred_result.get('confidence', 0.0))
            is_mock = bool(pred_result.get('is_mock', False))
            status = 'AUTO_ACCEPTED' if (conf >= threshold and not is_mock) else 'NEEDS_REVIEW'
            label = pred_result.get('predicted_label') or pred_result.get('label', 'notumor')

            prediction = Prediction.objects.create(
                scan=scan,
                predicted_label=label,
                confidence=conf,
                probabilities=pred_result.get('probabilities', {}),
                model_name=pred_result.get('model_name', 'vgg16'),
                is_mock=is_mock,
                status=status
            )

            # Attach Grad-CAM heatmap if generated
            heatmap_path = pred_result.get('heatmap_path')
            if heatmap_path:
                if not os.path.isabs(heatmap_path):
                    full_heat_path = os.path.join(settings.MEDIA_ROOT, heatmap_path)
                else:
                    full_heat_path = heatmap_path

                if os.path.exists(full_heat_path):
                    # Relative path within MEDIA_ROOT can be assigned directly
                    rel_path = os.path.relpath(full_heat_path, settings.MEDIA_ROOT).replace('\\', '/')
                    prediction.heatmap_image = rel_path
                    prediction.save(update_fields=['heatmap_image'])

            messages.success(
                request,
                f"Scan #{scan.id} uploaded and analyzed: {prediction.predicted_label.capitalize()} "
                f"({prediction.confidence:.1%} confidence)."
            )
            return redirect('scans:result', pk=scan.pk)
    else:
        form = ScanUploadForm(initial=initial_data, user=request.user)

    return render(request, 'scans/scan_upload.html', {'form': form})


@login_required
def scan_result(request, pk):
    """Display scan details, AI classification, Grad-CAM visualization, and clinical review."""
    scan = get_object_or_404(MRIScan, pk=pk)

    if not can_access_patient(request.user, scan.patient):
        messages.error(request, 'Access denied. You are not assigned to this patient.')
        return redirect('scans:dashboard')

    prediction = getattr(scan, 'prediction', None)
    review = getattr(prediction, 'review', None) if prediction else None

    # Calculate probabilities sorted descending for display
    sorted_probs = []
    if prediction and prediction.probabilities:
        sorted_probs = sorted(
            prediction.probabilities.items(),
            key=lambda item: item[1],
            reverse=True
        )

    context = {
        'scan': scan,
        'prediction': prediction,
        'review': review,
        'sorted_probs': sorted_probs,
        'threshold': getattr(settings, 'NEURO_CONFIDENCE_THRESHOLD', 0.80),
    }
    return render(request, 'scans/scan_result.html', context)


@login_required
def review_queue(request):
    """List pending predictions requiring human clinician review, ordered by lowest confidence."""
    user = request.user
    is_admin = (hasattr(user, 'profile') and user.profile.is_admin) or user.is_superuser

    if is_admin:
        pending_qs = Prediction.objects.filter(status='NEEDS_REVIEW')
    else:
        pending_qs = Prediction.objects.filter(
            status='NEEDS_REVIEW',
            scan__patient__care_team__doctor=user
        ).distinct()

    pending_qs = pending_qs.select_related('scan', 'scan__patient', 'scan__uploaded_by').order_by('confidence')

    return render(request, 'scans/review_queue.html', {
        'predictions': pending_qs,
        'count': pending_qs.count(),
    })


@login_required
def review_scan(request, pk):
    """Clinical review form for a doctor to confirm or override an AI prediction."""
    # Allow pk to be either Prediction.pk or MRIScan.pk
    try:
        prediction = Prediction.objects.select_related('scan', 'scan__patient').get(pk=pk)
    except Prediction.DoesNotExist:
        scan = get_object_or_404(MRIScan, pk=pk)
        prediction = get_object_or_404(Prediction, scan=scan)

    if not can_access_patient(request.user, prediction.scan.patient):
        messages.error(request, 'Access denied. You cannot review scans for this patient.')
        return redirect('scans:review_queue')

    existing_review = getattr(prediction, 'review', None)

    if request.method == 'POST':
        form = ReviewForm(request.POST)
        if form.is_valid():
            final_label = form.cleaned_data['final_label']
            agrees = form.cleaned_data['agrees_with_ai']
            notes = form.cleaned_data['clinical_notes']

            DoctorReview.objects.update_or_create(
                prediction=prediction,
                defaults={
                    'reviewer': request.user,
                    'final_label': final_label,
                    'agrees_with_ai': agrees,
                    'clinical_notes': notes,
                }
            )

            prediction.status = 'REVIEWED'
            prediction.save()

            messages.success(request, f'Review submitted successfully for Scan #{prediction.scan.id}.')
            return redirect('scans:result', pk=prediction.scan.pk)
    else:
        initial = {}
        if existing_review:
            initial = {
                'final_label': existing_review.final_label,
                'agrees_with_ai': existing_review.agrees_with_ai,
                'clinical_notes': existing_review.clinical_notes,
            }
        else:
            initial = {
                'final_label': prediction.predicted_label,
                'agrees_with_ai': True,
            }
        form = ReviewForm(initial=initial)

    return render(request, 'scans/review_form.html', {
        'prediction': prediction,
        'scan': prediction.scan,
        'form': form,
        'review': existing_review,
    })


@login_required
def search_api(request):
    """Live JSON search endpoint for patients and scans."""
    query = request.GET.get('q', '').strip()
    if not query:
        return JsonResponse({'patients': [], 'scans': []})

    user = request.user
    is_admin = (hasattr(user, 'profile') and user.profile.is_admin) or user.is_superuser

    if is_admin:
        patients_qs = Patient.objects.all()
        scans_qs = MRIScan.objects.all()
    else:
        patients_qs = Patient.objects.filter(care_team__doctor=user).distinct()
        scans_qs = MRIScan.objects.filter(patient__care_team__doctor=user).distinct()

    matched_patients = patients_qs.filter(
        Q(patient_code__icontains=query) |
        Q(full_name__icontains=query) |
        Q(contact__icontains=query)
    )[:6]

    matched_scans = scans_qs.filter(
        Q(patient__patient_code__icontains=query) |
        Q(patient__full_name__icontains=query) |
        Q(prediction__predicted_label__icontains=query)
    ).select_related('patient', 'prediction')[:6]

    patient_results = [
        {
            'id': p.id,
            'code': p.patient_code,
            'name': p.full_name,
            'url': f"/patients/detail/{p.id}/",
        }
        for p in matched_patients
    ]

    scan_results = [
        {
            'id': s.id,
            'patient_code': s.patient.patient_code,
            'patient_name': s.patient.full_name,
            'predicted_label': s.prediction.predicted_label if hasattr(s, 'prediction') else 'Pending',
            'confidence': f"{s.prediction.confidence:.1%}" if hasattr(s, 'prediction') else '',
            'url': f"/scans/result/{s.id}/",
        }
        for s in matched_scans
    ]

    return JsonResponse({
        'patients': patient_results,
        'scans': scan_results,
    })
