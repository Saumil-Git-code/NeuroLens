/**
 * NeuroLens Analytics Charts Module
 * Initializes Chart.js visualizations using data fetched from /analytics/api/data/
 */

document.addEventListener('DOMContentLoaded', function () {
  const classDistCanvas = document.getElementById('classDistributionChart');
  const scansPerDayCanvas = document.getElementById('scansPerDayChart');
  const confidenceCanvas = document.getElementById('confidenceHistChart');

  if (!classDistCanvas && !scansPerDayCanvas && !confidenceCanvas) {
    return;
  }

  fetch('/analytics/api/data/', {
    headers: {
      'Accept': 'application/json',
      'X-Requested-With': 'XMLHttpRequest'
    }
  })
    .then(response => {
      if (!response.ok) {
        throw new Error('Analytics API response error');
      }
      return response.json();
    })
    .then(data => {
      renderClassDistribution(data.class_distribution);
      renderScansPerDay(data.scans_per_day);
      renderConfidenceHistogram(data.confidence_histogram);
      renderAgreementTable(data.review_table);
    })
    .catch(error => {
      console.error('Error loading analytics data:', error);
    });

  function renderClassDistribution(dist) {
    if (!classDistCanvas || !dist) return;

    const labels = Object.keys(dist).map(l => l.charAt(0).toUpperCase() + l.slice(1));
    const values = Object.values(dist);
    const total = values.reduce((a, b) => a + b, 0);

    if (total === 0) {
      showEmptyChartMessage(classDistCanvas, 'No MRI scan predictions recorded yet');
      return;
    }

    new Chart(classDistCanvas, {
      type: 'doughnut',
      data: {
        labels: labels,
        datasets: [{
          data: values,
          backgroundColor: [
            '#A65D5D', // Glioma
            '#A98245', // Meningioma
            '#4F806A', // No Tumor
            '#54748D', // Pituitary
          ],
          borderWidth: 2,
          borderColor: '#FFFFFF',
          hoverOffset: 6
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'bottom',
            labels: {
              boxWidth: 12,
              font: { family: 'Inter', size: 12 },
              padding: 16
            }
          },
          tooltip: {
            callbacks: {
              label: function (context) {
                const val = context.raw || 0;
                const pct = total > 0 ? ((val / total) * 100).toFixed(1) : 0;
                return ` ${context.label}: ${val} scans (${pct}%)`;
              }
            }
          }
        },
        cutout: '68%'
      }
    });
  }

  function renderScansPerDay(dailyList) {
    if (!scansPerDayCanvas || !dailyList) return;

    const labels = dailyList.map(item => item.date);
    const dataPoints = dailyList.map(item => item.count);

    new Chart(scansPerDayCanvas, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [{
          label: 'Scans Processed',
          data: dataPoints,
          borderColor: '#285E61',
          backgroundColor: 'rgba(40, 94, 97, 0.08)',
          fill: true,
          tension: 0.35,
          borderWidth: 2.5,
          pointRadius: 3,
          pointHoverRadius: 6,
          pointBackgroundColor: '#285E61'
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: function (ctx) {
                return ` ${ctx.parsed.y} scan(s)`;
              }
            }
          }
        },
        scales: {
          y: {
            beginAtZero: true,
            ticks: {
              stepSize: 1,
              font: { family: 'Inter', size: 11 }
            },
            grid: { color: '#F1F5F9' }
          },
          x: {
            ticks: {
              maxRotation: 45,
              minRotation: 0,
              font: { family: 'Inter', size: 10 }
            },
            grid: { display: false }
          }
        }
      }
    });
  }

  function renderConfidenceHistogram(hist) {
    if (!confidenceCanvas || !hist) return;

    const total = hist.counts.reduce((a, b) => a + b, 0);
    if (total === 0) {
      showEmptyChartMessage(confidenceCanvas, 'No confidence records available');
      return;
    }

    new Chart(confidenceCanvas, {
      type: 'bar',
      data: {
        labels: hist.labels,
        datasets: [{
          label: 'Prediction Count',
          data: hist.counts,
          backgroundColor: [
            '#D9B7B7', // 0-20%
            '#DDC99E', // 20-40%
            '#E5D9B8', // 40-60%
            '#B7CBBE', // 60-80%
            '#285E61', // 80-100%
          ],
          borderRadius: 6,
          borderSkipped: false
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: function (ctx) {
                return ` ${ctx.parsed.y} scans in this confidence bracket`;
              }
            }
          }
        },
        scales: {
          y: {
            beginAtZero: true,
            ticks: {
              stepSize: 1,
              font: { family: 'Inter', size: 11 }
            },
            grid: { color: '#F1F5F9' }
          },
          x: {
            ticks: { font: { family: 'Inter', size: 11 } },
            grid: { display: false }
          }
        }
      }
    });
  }

  function renderAgreementTable(reviewList) {
    const tableBody = document.getElementById('agreementTableBody');
    if (!tableBody || !reviewList) return;

    if (reviewList.length === 0) {
      tableBody.innerHTML = `
        <tr>
          <td colspan="6" class="text-center py-4 text-muted">
            <i class="bi bi-info-circle me-1"></i> No doctor reviews submitted yet.
          </td>
        </tr>
      `;
      return;
    }

    let html = '';
    reviewList.forEach(item => {
      const agreeBadge = item.agrees
        ? '<span class="badge bg-success-subtle text-success border border-success-subtle"><i class="bi bi-check-circle me-1"></i> Agreed</span>'
        : '<span class="badge bg-warning-subtle text-warning border border-warning-subtle"><i class="bi bi-arrow-repeat me-1"></i> Override</span>';

      html += `
        <tr>
          <td><span class="fw-semibold text-primary">${item.patient_code}</span></td>
          <td>${item.doctor_name}</td>
          <td><span class="badge bg-light text-dark border">${item.ai_label}</span></td>
          <td><strong>${item.doctor_label}</strong></td>
          <td>${agreeBadge}</td>
          <td class="text-muted small">${item.date}</td>
        </tr>
      `;
    });
    tableBody.innerHTML = html;
  }

  function showEmptyChartMessage(canvas, message) {
    const parent = canvas.parentElement;
    canvas.style.display = 'none';
    const msgDiv = document.createElement('div');
    msgDiv.className = 'd-flex align-items-center justify-content-center h-100 text-muted small py-5';
    msgDiv.innerHTML = `<i class="bi bi-bar-chart me-2"></i> ${message}`;
    parent.appendChild(msgDiv);
  }
});
