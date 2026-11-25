from datetime import timedelta
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.utils import timezone
from django.db.models import Count
from django.db.models.functions import TruncDate

from scans.models import MRIScan, Prediction, DoctorReview
from patients.models import Patient
from ml import CLASS_NAMES


def get_filtered_querysets(user):
    """Helper to return role-filtered querysets for patients, scans, and predictions."""
    is_admin = (hasattr(user, 'profile') and user.profile.is_admin) or user.is_superuser
    if is_admin:
        patients = Patient.objects.all()
        scans = MRIScan.objects.all()
        predictions = Prediction.objects.all()
        reviews = DoctorReview.objects.all()
    else:
        patients = Patient.objects.filter(care_team__doctor=user).distinct()
        scans = MRIScan.objects.filter(patient__care_team__doctor=user).distinct()
        predictions = Prediction.objects.filter(scan__patient__care_team__doctor=user).distinct()
        reviews = DoctorReview.objects.filter(prediction__scan__patient__care_team__doctor=user).distinct()

    return patients, scans, predictions, reviews


@login_required
def analytics_dashboard(request):
    """Render the analytics dashboard with Chart.js visualization containers."""
    patients, scans, predictions, reviews = get_filtered_querysets(request.user)

    total_patients = patients.count()
    total_scans = scans.count()
    pending_count = predictions.filter(status='NEEDS_REVIEW').count()
    total_reviews = reviews.count()
    agreed_reviews = reviews.filter(agrees_with_ai=True).count()
    agreement_rate = (agreed_reviews / total_reviews * 100) if total_reviews > 0 else 0.0

    context = {
        'total_patients': total_patients,
        'total_scans': total_scans,
        'pending_count': pending_count,
        'agreement_rate': round(agreement_rate, 1),
        'total_reviews': total_reviews,
        'agreed_reviews': agreed_reviews,
        'disagreed_reviews': total_reviews - agreed_reviews,
    }
    return render(request, 'analytics/dashboard.html', context)


@login_required
def analytics_data_api(request):
    """JSON API endpoint providing data feeds for Chart.js charts."""
    patients, scans, predictions, reviews = get_filtered_querysets(request.user)

    # 1. Class Distribution
    class_dist = {cls_name: 0 for cls_name in CLASS_NAMES}
    pred_counts = predictions.values('predicted_label').annotate(count=Count('id'))
    for item in pred_counts:
        lbl = item['predicted_label'].lower()
        if lbl in class_dist:
            class_dist[lbl] = item['count']
        else:
            class_dist[lbl] = item['count']

    # 2. Scans per Day (Last 30 Days)
    today = timezone.now().date()
    start_date = today - timedelta(days=29)
    daily_counts_qs = (
        scans.filter(uploaded_at__date__gte=start_date)
        .annotate(day=TruncDate('uploaded_at'))
        .values('day')
        .annotate(count=Count('id'))
    )
    daily_map = {item['day']: item['count'] for item in daily_counts_qs}

    scans_per_day = []
    for day_offset in range(30):
        d = start_date + timedelta(days=day_offset)
        scans_per_day.append({
            'date': d.strftime('%b %d'),
            'count': daily_map.get(d, 0)
        })

    # 3. Confidence Histogram (5 Bins)
    # Bins: [0.0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0]
    bins = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
    hist_counts = [0, 0, 0, 0, 0]
    for conf in predictions.values_list('confidence', flat=True):
        if conf is None:
            continue
        if conf < 0.2:
            hist_counts[0] += 1
        elif conf < 0.4:
            hist_counts[1] += 1
        elif conf < 0.6:
            hist_counts[2] += 1
        elif conf < 0.8:
            hist_counts[3] += 1
        else:
            hist_counts[4] += 1

    confidence_histogram = {
        'labels': bins,
        'counts': hist_counts
    }

    # 4. Human-AI Agreement Stats
    total_reviews = reviews.count()
    agreed_count = reviews.filter(agrees_with_ai=True).count()
    disagreed_count = total_reviews - agreed_count
    rate = (agreed_count / total_reviews * 100) if total_reviews > 0 else 0.0

    agreement_data = {
        'total_reviews': total_reviews,
        'agreed': agreed_count,
        'disagreed': disagreed_count,
        'agreement_rate': round(rate, 1),
    }

    # 5. Review Details for Agreement Table
    recent_reviews = reviews.select_related('prediction', 'prediction__scan', 'prediction__scan__patient', 'reviewer').order_by('-reviewed_at')[:8]
    review_table = [
        {
            'patient_code': r.prediction.scan.patient.patient_code,
            'ai_label': r.prediction.predicted_label.capitalize(),
            'doctor_label': r.final_label.capitalize(),
            'agrees': r.agrees_with_ai,
            'doctor_name': f"Dr. {r.reviewer.get_full_name() or r.reviewer.username}" if r.reviewer else 'Physician',
            'date': r.reviewed_at.strftime('%Y-%m-%d'),
        }
        for r in recent_reviews
    ]

    return JsonResponse({
        'class_distribution': class_dist,
        'scans_per_day': scans_per_day,
        'confidence_histogram': confidence_histogram,
        'agreement': agreement_data,
        'review_table': review_table,
        'stats': {
            'total_patients': patients.count(),
            'total_scans': scans.count(),
            'pending_reviews': predictions.filter(status='NEEDS_REVIEW').count(),
            'agreement_rate': round(rate, 1),
        }
    })
