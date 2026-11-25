/**
 * NeuroLens Live Search Module
 * Provides debounced real-time search across patients and MRI scans.
 */

document.addEventListener('DOMContentLoaded', function () {
  const searchInput = document.getElementById('globalSearchInput');
  const searchResults = document.getElementById('searchResultsDropdown');

  if (!searchInput || !searchResults) {
    return;
  }

  let debounceTimer = null;

  searchInput.addEventListener('input', function () {
    const query = this.value.trim();

    clearTimeout(debounceTimer);

    if (query.length < 2) {
      searchResults.classList.remove('show');
      searchResults.innerHTML = '';
      return;
    }

    debounceTimer = setTimeout(() => {
      fetchSearchResults(query);
    }, 300);
  });

  function fetchSearchResults(query) {
    fetch(`/scans/api/search/?q=${encodeURIComponent(query)}`, {
      headers: {
        'Accept': 'application/json',
        'X-Requested-With': 'XMLHttpRequest',
      }
    })
      .then(response => {
        if (!response.ok) {
          throw new Error('Search network error');
        }
        return response.json();
      })
      .then(data => {
        renderResults(data, query);
      })
      .catch(error => {
        console.error('Live search error:', error);
      });
  }

  function renderResults(data, query) {
    searchResults.innerHTML = '';

    const hasPatients = data.patients && data.patients.length > 0;
    const hasScans = data.scans && data.scans.length > 0;

    if (!hasPatients && !hasScans) {
      searchResults.innerHTML = `
        <div class="p-3 text-center text-muted small">
          <i class="bi bi-search me-1"></i> No matching records found for "${escapeHtml(query)}"
        </div>
      `;
      searchResults.classList.add('show');
      return;
    }

    let html = '';

    if (hasPatients) {
      html += '<div class="search-category-header"><i class="bi bi-people me-1"></i> Patients</div>';
      data.patients.forEach(patient => {
        html += `
          <a href="${patient.url}" class="search-item">
            <div class="fw-semibold text-primary">${escapeHtml(patient.name)}</div>
            <div class="text-muted small">Code: ${escapeHtml(patient.code)}</div>
          </a>
        `;
      });
    }

    if (hasScans) {
      html += '<div class="search-category-header"><i class="bi bi-images me-1"></i> Scans & Predictions</div>';
      data.scans.forEach(scan => {
        html += `
          <a href="${scan.url}" class="search-item">
            <div class="d-flex justify-content-between align-items-center">
              <span class="fw-semibold">Scan #${scan.id} (${escapeHtml(scan.patient_code)})</span>
              <span class="badge bg-light text-dark border">${escapeHtml(scan.predicted_label)} ${scan.confidence}</span>
            </div>
            <div class="text-muted small">${escapeHtml(scan.patient_name)}</div>
          </a>
        `;
      });
    }

    searchResults.innerHTML = html;
    searchResults.classList.add('show');
  }

  // Close dropdown on click outside
  document.addEventListener('click', function (event) {
    if (!searchInput.contains(event.target) && !searchResults.contains(event.target)) {
      searchResults.classList.remove('show');
    }
  });

  function escapeHtml(text) {
    if (!text) return '';
    return String(text)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }
});
