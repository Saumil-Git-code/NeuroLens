"""Context processors for NeuroLens."""


def pending_reviews_count(request):
    """Context processor providing count of pending reviews for badges."""
    if request.user.is_authenticated:
        try:
            from scans.models import Prediction
            if hasattr(request.user, 'profile') and request.user.profile.is_doctor:
                count = Prediction.objects.filter(
                    status='NEEDS_REVIEW',
                    scan__patient__care_team__doctor=request.user
                ).distinct().count()
            else:
                count = Prediction.objects.filter(status='NEEDS_REVIEW').count()
            return {'pending_reviews_count': count}
        except Exception:
            return {'pending_reviews_count': 0}
    return {'pending_reviews_count': 0}
