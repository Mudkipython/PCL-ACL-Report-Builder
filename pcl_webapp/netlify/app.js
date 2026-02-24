const form = document.getElementById('upload-form');
const errorEl = document.getElementById('error');
const statusEl = document.getElementById('status');

const apiBase = document
  .querySelector('meta[name="api-base"]')
  .getAttribute('content')
  .replace(/\/$/, '');

function showError(message) {
  errorEl.textContent = message;
  errorEl.classList.remove('hidden');
}

function clearError() {
  errorEl.textContent = '';
  errorEl.classList.add('hidden');
}

function showStatus(message) {
  statusEl.textContent = message;
  statusEl.classList.remove('hidden');
}

function hideStatus() {
  statusEl.textContent = '';
  statusEl.classList.add('hidden');
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  clearError();

  const formData = new FormData(form);

  if (!apiBase || apiBase.includes('YOUR-FASTAPI-DOMAIN')) {
    showError('Set your backend URL in the meta api-base tag.');
    return;
  }

  try {
    showStatus('Processing...');
    const response = await fetch(`${apiBase}/process`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const text = await response.text();
      showError(text || 'Failed to generate report.');
      return;
    }

    const disposition = response.headers.get('content-disposition') || '';
    const match = disposition.match(/filename="?([^";]+)"?/i);
    const filename = match ? match[1] : 'pcl_report';

    const blob = await response.blob();
    const url = window.URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = filename;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    window.URL.revokeObjectURL(url);
  } catch (err) {
    showError('Unexpected error while uploading.');
  } finally {
    hideStatus();
  }
});
