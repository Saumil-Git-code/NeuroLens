import csv
import io
import os
from django.shortcuts import get_object_or_404, redirect, render
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import HttpResponse, Http404
from django.utils import timezone

from scans.models import MRIScan, Prediction, DoctorReview
from patients.models import Patient
from patients.views import can_access_patient


@login_required
def export_options(request):
    """Render export options and reports management hub."""
    return render(request, 'reports/export_options.html')


@login_required
def export_csv(request):
    """Export filtered scans and prediction records as a CSV spreadsheet."""
    user = request.user
    is_admin = (hasattr(user, 'profile') and user.profile.is_admin) or user.is_superuser

    if is_admin:
        scans = MRIScan.objects.select_related('patient', 'prediction', 'uploaded_by').all()
    else:
        scans = MRIScan.objects.filter(
            patient__care_team__doctor=user
        ).select_related('patient', 'prediction', 'uploaded_by').distinct()

    # Optional query filters
    status = request.GET.get('status')
    label = request.GET.get('label')
    if status:
        scans = scans.filter(prediction__status=status)
    if label:
        scans = scans.filter(prediction__predicted_label=label)

    response = HttpResponse(content_type='text/csv')
    timestamp = timezone.now().strftime('%Y%m%d_%H%M%S')
    response['Content-Disposition'] = f'attachment; filename="neurolens_scans_export_{timestamp}.csv"'

    writer = csv.writer(response)
    writer.writerow([
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
    ])

    for scan in scans:
        pred = getattr(scan, 'prediction', None)
        review = getattr(pred, 'review', None) if pred else None
        uploaded_by_name = scan.uploaded_by.get_full_name() if scan.uploaded_by else 'N/A'

        writer.writerow([
            scan.id,
            scan.patient.patient_code,
            scan.patient.full_name,
            scan.patient.age,
            scan.patient.get_gender_display(),
            scan.uploaded_at.strftime('%Y-%m-%d %H:%M:%S'),
            uploaded_by_name,
            pred.predicted_label if pred else 'N/A',
            f"{pred.confidence:.4f}" if pred else 'N/A',
            pred.status if pred else 'N/A',
            pred.model_name if pred else 'N/A',
            'Yes' if (pred and pred.is_mock) else 'No',
            'Yes' if review else 'No',
            review.final_label if review else 'N/A',
            ('Yes' if review.agrees_with_ai else 'No') if review else 'N/A',
            review.clinical_notes if review else '',
        ])

    return response


@login_required
def generate_pdf(request, scan_id):
    """Generate a downloadable PDF clinical report for an MRI scan using ReportLab."""
    scan = get_object_or_404(MRIScan, pk=scan_id)

    if not can_access_patient(request.user, scan.patient):
        messages.error(request, 'Access denied. You cannot view reports for this patient.')
        return redirect('scans:dashboard')

    prediction = getattr(scan, 'prediction', None)
    review = getattr(prediction, 'review', None) if prediction else None

    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, KeepTogether
    except ImportError:
        # Fallback if ReportLab not found
        return HttpResponse('ReportLab library is not installed.', status=500)

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    story = []
    styles = getSampleStyleSheet()

    # Custom Color Palette
    teal_primary = colors.HexColor('#0F766E')
    teal_dark = colors.HexColor('#115E59')
    gray_bg = colors.HexColor('#F8FAFC')
    gray_border = colors.HexColor('#CBD5E1')
    red_warn = colors.HexColor('#DC2626')

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=20,
        leading=24,
        textColor=teal_primary,
        spaceAfter=6,
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor('#64748B'),
        spaceAfter=14,
    )
    section_heading = ParagraphStyle(
        'SectionHead',
        parent=styles['Heading2'],
        fontSize=12,
        leading=16,
        textColor=teal_dark,
        spaceBefore=10,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#1E293B'),
    )
    bold_style = ParagraphStyle(
        'BodyBold',
        parent=body_style,
        fontName='Helvetica-Bold',
    )
    warning_style = ParagraphStyle(
        'WarningStyle',
        parent=body_style,
        textColor=red_warn,
        fontName='Helvetica-Bold',
    )
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=body_style,
        fontSize=8,
        leading=10,
        textColor=colors.HexColor('#64748B'),
    )

    # 1. Header Banner
    story.append(Paragraph("NeuroLens &bull; AI Brain MRI Clinical Report", title_style))
    story.append(Paragraph(
        f"Generated on {timezone.now().strftime('%B %d, %Y at %H:%M UTC')} | Reference ID: NL-SCAN-{scan.id:05d}",
        subtitle_style
    ))
    story.append(Spacer(1, 4))

    # 2. Demo Mode Alert (if mock prediction)
    if prediction and prediction.is_mock:
        demo_alert = [
            [Paragraph("DEMO MODE NOTICE: This prediction was generated using synthetic baseline weights. Not for diagnostic use.", warning_style)]
        ]
        t_demo = Table(demo_alert, colWidths=[540])
        t_demo.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#FEF2F2')),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#F87171')),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(t_demo)
        story.append(Spacer(1, 8))

    # 3. Patient Demographics & Scan Information Table
    story.append(Paragraph("Patient Demographics & Scan Metadata", section_heading))
    patient = scan.patient
    patient_info_data = [
        [Paragraph("<b>Patient Code:</b>", body_style), Paragraph(patient.patient_code, body_style),
         Paragraph("<b>Full Name:</b>", body_style), Paragraph(patient.full_name, body_style)],
        [Paragraph("<b>Age:</b>", body_style), Paragraph(f"{patient.age} years", body_style),
         Paragraph("<b>Gender:</b>", body_style), Paragraph(patient.get_gender_display(), body_style)],
        [Paragraph("<b>Scan Date:</b>", body_style), Paragraph(scan.uploaded_at.strftime('%Y-%m-%d %H:%M'), body_style),
         Paragraph("<b>Uploaded By:</b>", body_style), Paragraph(scan.uploaded_by.get_full_name() if scan.uploaded_by else 'Staff', body_style)],
        [Paragraph("<b>Clinical Notes:</b>", body_style), Paragraph(scan.notes or 'None provided', body_style),
         Paragraph("<b>Patient Contact:</b>", body_style), Paragraph(patient.contact or 'N/A', body_style)],
    ]
    t_patient = Table(patient_info_data, colWidths=[100, 170, 100, 170])
    t_patient.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), gray_bg),
        ('BOX', (0, 0), (-1, -1), 0.5, gray_border),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, gray_border),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t_patient)
    story.append(Spacer(1, 10))

    # 4. AI Tumor Classification Findings
    story.append(Paragraph("AI Classification Findings", section_heading))
    if prediction:
        pred_label_str = prediction.predicted_label.capitalize()
        status_str = prediction.get_status_display()
        conf_str = f"{prediction.confidence:.1%}"

        pred_data = [
            [Paragraph("<b>Primary Prediction:</b>", body_style), Paragraph(f"<b>{pred_label_str}</b>", body_style),
             Paragraph("<b>Confidence Score:</b>", body_style), Paragraph(f"<b>{conf_str}</b>", body_style)],
            [Paragraph("<b>Model Architecture:</b>", body_style), Paragraph(prediction.model_name.upper(), body_style),
             Paragraph("<b>Triaged Status:</b>", body_style), Paragraph(status_str, body_style)],
        ]
        t_pred = Table(pred_data, colWidths=[120, 150, 120, 150])
        t_pred.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F0FDFA')),
            ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#99F6E4')),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CCFBF1')),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(t_pred)

        # Probabilities breakdown table
        if prediction.probabilities:
            story.append(Spacer(1, 6))
            prob_rows = [[Paragraph(f"<b>{k.capitalize()}</b>", body_style), f"{v:.1%}"] for k, v in prediction.probabilities.items()]
            t_prob = Table(
                [[Paragraph("<b>Differential Class</b>", bold_style), Paragraph("<b>Probability</b>", bold_style)]] + prob_rows,
                colWidths=[270, 270]
            )
            t_prob.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E2E8F0')),
                ('BOX', (0, 0), (-1, -1), 0.5, gray_border),
                ('INNERGRID', (0, 0), (-1, -1), 0.5, gray_border),
                ('TOPPADDING', (0, 0), (-1, -1), 3),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ]))
            story.append(t_prob)
    else:
        story.append(Paragraph("No prediction generated yet for this scan.", body_style))

    story.append(Spacer(1, 10))

    # 5. Visual Imagery (MRI + Grad-CAM Heatmap side-by-side)
    image_cells = []
    has_mri = False
    has_heatmap = False

    if scan.image and os.path.exists(scan.image.path):
        try:
            mri_img = RLImage(scan.image.path, width=2.4 * 72, height=2.4 * 72)
            has_mri = True
        except Exception:
            mri_img = Paragraph("Unable to load MRI image", body_style)
    else:
        mri_img = Paragraph("MRI image not available", body_style)

    if prediction and prediction.heatmap_image and os.path.exists(prediction.heatmap_image.path):
        try:
            heat_img = RLImage(prediction.heatmap_image.path, width=2.4 * 72, height=2.4 * 72)
            has_heatmap = True
        except Exception:
            heat_img = Paragraph("Grad-CAM visualization not generated", body_style)
    else:
        heat_img = Paragraph("Grad-CAM Heatmap not available", body_style)

    images_table_data = [
        [Paragraph("<b>Original Brain MRI</b>", body_style), Paragraph("<b>Grad-CAM Explainability Heatmap</b>", body_style)],
        [mri_img, heat_img]
    ]
    t_images = Table(images_table_data, colWidths=[270, 270])
    t_images.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOX', (0, 0), (-1, -1), 0.5, gray_border),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, gray_border),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(KeepTogether([
        Paragraph("Imaging & Visual AI Explainability", section_heading),
        t_images
    ]))
    story.append(Spacer(1, 10))

    # 6. Clinician Review Section
    story.append(Paragraph("Physician Review & Clinical Sign-off", section_heading))
    if review:
        doctor_name = review.reviewer.get_full_name() if review.reviewer else 'Clinical Doctor'
        agreement_text = "Concurs with AI prediction" if review.agrees_with_ai else "Disagreed with AI prediction (Manual Override)"
        review_data = [
            [Paragraph("<b>Reviewing Clinician:</b>", body_style), Paragraph(f"Dr. {doctor_name}", body_style),
             Paragraph("<b>Review Date:</b>", body_style), Paragraph(review.reviewed_at.strftime('%Y-%m-%d %H:%M'), body_style)],
            [Paragraph("<b>Final Clinical Label:</b>", body_style), Paragraph(f"<b>{review.final_label.capitalize()}</b>", body_style),
             Paragraph("<b>AI Alignment:</b>", body_style), Paragraph(agreement_text, body_style)],
            [Paragraph("<b>Clinical Notes & Rationale:</b>", body_style), Paragraph(review.clinical_notes or 'No additional notes recorded.', body_style),
             Paragraph("", body_style), Paragraph("", body_style)],
        ]
        t_review = Table(review_data, colWidths=[130, 140, 110, 160])
        t_review.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F1F5F9')),
            ('BOX', (0, 0), (-1, -1), 0.5, gray_border),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, gray_border),
            ('SPAN', (1, 2), (3, 2)),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(t_review)
    else:
        story.append(Paragraph("<i>Pending physician review in the clinical queue.</i>", body_style))

    story.append(Spacer(1, 14))

    # 7. Regulatory & Safety Disclaimer
    disclaimer_text = (
        "<b>MEDICAL DISCLAIMER:</b> NeuroLens is an assistive clinical decision-support tool powered by deep learning "
        "algorithms. It is NOT an autonomous medical diagnostic device and does not replace the professional judgment "
        "of a licensed physician, neuroradiologist, or neurosurgeon. All diagnostic and therapeutic interventions must "
        "be established through comprehensive clinical evaluations and institutional medical protocols."
    )
    story.append(Paragraph(disclaimer_text, disclaimer_style))

    doc.build(story)
    buffer.seek(0)

    response = HttpResponse(buffer.getvalue(), content_type='application/pdf')
    response['Content-Disposition'] = f'inline; filename="NeuroLens_Report_Scan_{scan.id}.pdf"'
    return response
